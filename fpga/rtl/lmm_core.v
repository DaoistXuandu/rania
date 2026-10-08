// LMM core: tiled int16 GEMM  Y[M,N] = round((X[M,K] . W[N,K]^T) / 2^10)  as int32.
// Arguments arrive already validated by lmm_top; this block only schedules, computes, scrubs.
//
// Dataflow (per request):
//   for each X block of TM=128 rows:      X block -> xbuf (R banks, row m in bank m%R)
//     for each W panel of TN=32 rows:     panel p in wbuf half p%2 while panel p+1 loads into
//                                         the other half (ping-pong, C banks, row n in bank n%C)
//       for each row tile rt (R rows), col tile ct (C cols), k-word kw (4 int16):
//         PE(i,j) += dot4(xbuf[i][rt,kw], wbuf[j][ct,kw])      R*C*4 = 128 MAC/cycle
//       after the last kw: round+shift, drain 16 words into obuf half `ob`; after the last
//       col tile of a row tile the R x TN segment is handed to the writer (ping-pong obuf).
// After DONE or abort every buffer and register holding data is zeroed (scrub).
module lmm_core #(parameter R = 4, C = 8, TM = 128, TN = 32, KWMAX = 192) (
    input             clk, rst,
    input             go, abort,
    input      [10:0] M,
    input      [9:0]  K,
    input      [11:0] N,
    input      [31:0] xa, wa, ya,
    input      [31:0] x_lo, x_hi, w_lo, w_hi, y_lo, y_hi,
    output reg        finished,          // pulse; ok says whether Y is complete
    output reg        ok,
    output            fault,             // guard layer 2 tripped
    output            busy,
    output            mac_active,
    // read-only port (X, W)
    output     [31:0] rd_address,
    output            rd_read,
    output     [3:0]  rd_burstcount,
    input             rd_waitrequest,
    input      [63:0] rd_readdata,
    input             rd_readdatavalid,
    // bidirectional port (Y writes + fence read)
    output     [31:0] wr_address,
    output            wr_write,
    output            wr_read,
    output     [3:0]  wr_burstcount,
    output     [63:0] wr_writedata,
    input             wr_waitrequest,
    input             wr_readdatavalid
);
    localparam XD = TM / R * KWMAX, WH = TN / C * KWMAX, NCT = TN / C;
    localparam IDLE = 0, LDX = 1, LDW = 2, PANEL = 3, FLUSH = 4, FENCE = 5, STOP = 6, SCRUB = 7;
    reg [2:0] st;
    reg       kick;                          // reader was started this cycle; busy not yet visible

    wire [7:0]  KW   = K[9:2];               // 64-bit words per row
    wire [31:0] NB4  = {18'd0, N, 2'b0};     // Y row stride in bytes
    wire [31:0] K32  = {22'd0, K};

    // ---------------- request position ----------------
    reg  [10:0] rows_left;                   // rows of M not yet covered by finished blocks
    wire [7:0]  rows_b = rows_left > TM ? TM : rows_left[7:0];
    reg  [6:0]  p;                           // panel index
    reg  [31:0] xblk, yblk, wpan, ypan;
    wire        last_panel = p == N[11:5] - 1'b1;  // N/TN panels (TN = 32)

    // ---------------- scrub ----------------
    reg  [12:0] sc;
    wire        scrub = st == SCRUB;

    // ---------------- reader + loader mapping ----------------
    reg         rd_start, ld_w, ld_half;
    reg  [31:0] rd_addr_c;
    reg  [15:0] rd_nb_c;
    wire        rd_busy, rd_fault, beat_valid;
    wire [63:0] beat_data;
    reg  [7:0]  lkw;                         // word inside the row being loaded
    reg  [3:0]  lb;                          // destination bank
    reg  [12:0] lbase;                       // row base inside the bank
    wire        loading = beat_valid && st != STOP;

    lmm_rd u_rd (.clk(clk), .rst(rst), .clr(scrub), .start(rd_start), .addr(rd_addr_c), .nburst(rd_nb_c),
        .lo(ld_w ? w_lo : x_lo), .hi(ld_w ? w_hi : x_hi), .stop(st == STOP), .busy(rd_busy),
        .fault(rd_fault), .avm_address(rd_address), .avm_read(rd_read),
        .avm_burstcount(rd_burstcount), .avm_waitrequest(rd_waitrequest),
        .avm_readdata(rd_readdata), .avm_readdatavalid(rd_readdatavalid),
        .beat_valid(beat_valid), .beat_data(beat_data));

    // ---------------- compute issue (stage 0) ----------------
    reg         run, wsel, ob;
    reg  [5:0]  rt;
    reg  [1:0]  ct;
    reg  [7:0]  kw;
    reg  [12:0] xrt, wct;                    // rt*KW, ct*KW
    reg  [7:0]  rows_rt;                     // rows of this block not yet tiled
    reg  [31:0] yrt;
    reg  [1:0]  ready;                       // obuf half holds a finished segment
    reg  [63:0] meta_addr;
    reg  [5:0]  meta_rows;
    wire        seg_start = kw == 0 && ct == 0;
    wire        issue = run && !(seg_start && ready[ob]);
    wire        k_last = kw == KW - 1'b1;

    // pipeline tags: s1 = RAM data valid, s2 = dot products valid
    reg v1, f1, l1, v2, f2, l2;
    reg [1:0] ct1, ct2;
    reg ob1, ob2, se1, se2;

    // ---------------- buffers ----------------
    wire [64*R-1:0] xq;
    wire [64*C-1:0] wq;
    genvar gi, gj;
    generate
        for (gi = 0; gi < R; gi = gi + 1) begin : XB
            lmm_ram #(64, XD) ram (.clk(clk),
                .we(scrub || (loading && !ld_w && lb == gi)),
                .wa(scrub ? sc : lbase + lkw), .wd(scrub ? 64'd0 : beat_data),
                .re(issue), .ra(xrt + kw), .q(xq[64*gi +: 64]));
        end
        for (gj = 0; gj < C; gj = gj + 1) begin : WB
            wire [10:0] wa_ = scrub ? sc[10:0] : (ld_half ? WH : 0) + lbase + lkw;
            wire [10:0] ra_ = (wsel ? WH : 0) + wct + kw;
            lmm_ram #(64, 2 * WH) ram (.clk(clk),
                .we((scrub && sc < 2 * WH) || (loading && ld_w && lb == gj)),
                .wa(wa_), .wd(scrub ? 64'd0 : beat_data), .re(issue), .ra(ra_), .q(wq[64*gj +: 64]));
        end
    endgenerate

    // ---------------- PE array: dot4 -> accumulate (stage 2 -> 3) ----------------
    reg  signed [33:0] dot [0:R*C-1];
    reg  signed [41:0] acc [0:R*C-1];
    reg  [32*R*C-1:0]  drain;                // rounded results of the last finished tile
    generate
        for (gi = 0; gi < R; gi = gi + 1) begin : PR
            for (gj = 0; gj < C; gj = gj + 1) begin : PC
                wire signed [33:0] d =
                    $signed(xq[64*gi +  0 +: 16]) * $signed(wq[64*gj +  0 +: 16]) +
                    $signed(xq[64*gi + 16 +: 16]) * $signed(wq[64*gj + 16 +: 16]) +
                    $signed(xq[64*gi + 32 +: 16]) * $signed(wq[64*gj + 32 +: 16]) +
                    $signed(xq[64*gi + 48 +: 16]) * $signed(wq[64*gj + 48 +: 16]);
                wire signed [41:0] a = (f2 ? 42'sd0 : acc[gi*C+gj]) + dot[gi*C+gj];
                wire signed [41:0] r = (a + 42'sd512) >>> 10;
                always @(posedge clk) begin
                    if (scrub) begin
                        dot[gi*C+gj] <= 0; acc[gi*C+gj] <= 0; drain[32*(gi*C+gj) +: 32] <= 0;
                    end else begin
                        if (v1) dot[gi*C+gj] <= d;                  // clock enable = operand valid
                        if (v2) acc[gi*C+gj] <= a;
                        if (v2 && l2) drain[32*(gi*C+gj) +: 32] <= r[31:0];
                    end
                end
            end
        end
    endgenerate
    assign mac_active = v2;

    // ---------------- drain: 16 cycles, 2 int32 per obuf word ----------------
    reg        dact, d_ob, d_se;
    reg [1:0]  d_ct;
    reg [3:0]  dc;                           // {row i, word jj}
    wire [8:0] lo_i = dc[3:2] * C + {dc[1:0], 1'b0};   // element (row i, col 2*jj) of the tile
    wire [63:0] obuf_q;
    wire        obuf_re;
    wire [6:0]  obuf_ra;
    lmm_ram #(64, 4 * TN) u_obuf (.clk(clk),
        .we(scrub ? sc < 4 * TN : dact),
        .wa(scrub ? sc[6:0] : {d_ob, dc[3:2], d_ct, dc[1:0]}),
        .wd(scrub ? 64'd0 : {drain[32*(lo_i + 1) +: 32], drain[32*lo_i +: 32]}),
        .re(obuf_re), .ra(obuf_ra), .q(obuf_q));

    wire wr_busy, wr_fault, released, wr_wb, fence_done;
    reg  fence_req;
    lmm_wr u_wr (.clk(clk), .rst(rst), .clr(scrub), .lo(y_lo), .hi(y_hi), .row_stride(NB4),
        .ready(ready), .meta_addr(meta_addr), .meta_rows(meta_rows), .release_seg(released),
        .wb(wr_wb), .obuf_re(obuf_re), .obuf_ra(obuf_ra), .obuf_q(obuf_q), .stop(st == STOP),
        .fence_req(fence_req), .fence_done(fence_done), .fault(wr_fault), .busy(wr_busy),
        .avm_address(wr_address), .avm_write(wr_write), .avm_read(wr_read),
        .avm_burstcount(wr_burstcount), .avm_writedata(wr_writedata),
        .avm_waitrequest(wr_waitrequest), .avm_readdatavalid(wr_readdatavalid));

    assign fault = rd_fault || wr_fault;
    wire comp_busy = run || v1 || v2 || dact;
    assign busy = st != IDLE;

    // ---------------- control ----------------
    always @(posedge clk) begin
        rd_start <= 0; kick <= rd_start; finished <= 0; fence_req <= 0;
        if (rst) begin
            st <= IDLE; run <= 0; v1 <= 0; v2 <= 0; dact <= 0; ready <= 0; ok <= 0;
        end else begin
            // ---- loader address mapping: row-major beats -> (bank, row, word)
            if (loading) begin
                if (lkw == KW - 1'b1) begin
                    lkw <= 0;
                    if (lb == (ld_w ? C : R) - 1) begin lb <= 0; lbase <= lbase + KW; end
                    else lb <= lb + 1'b1;
                end else lkw <= lkw + 1'b1;
            end

            // ---- compute issue
            if (issue) begin
                if (seg_start) begin            // claim obuf half for this row tile
                    if (ob) begin meta_addr[63:32] <= yrt; meta_rows[5:3] <= rows_rt > R ? R : rows_rt[2:0]; end
                    else    begin meta_addr[31:0]  <= yrt; meta_rows[2:0] <= rows_rt > R ? R : rows_rt[2:0]; end
                end
                if (k_last) begin
                    kw <= 0;
                    if (ct == NCT - 1) begin
                        ct <= 0; wct <= 0; rt <= rt + 1'b1; xrt <= xrt + KW;
                        yrt <= yrt + (NB4 << 2); ob <= ~ob;           // + R rows
                        rows_rt <= rows_rt > R ? rows_rt - R : 8'd0;
                        if (rows_rt <= R) run <= 0;
                    end else begin
                        ct <= ct + 1'b1; wct <= wct + KW;
                    end
                end else kw <= kw + 1'b1;
            end
            v1 <= issue; f1 <= kw == 0; l1 <= k_last; ct1 <= ct; ob1 <= ob; se1 <= ct == NCT - 1;
            v2 <= v1;    f2 <= f1;      l2 <= l1;     ct2 <= ct1; ob2 <= ob1; se2 <= se1;

            // ---- drain finished tile into obuf; hand over the segment after its last tile
            // (K >= 64 keeps tiles >= 16 cycles apart, so a drain always ends before the next
            //  one starts, and a row tile takes >= 64 cycles, so a half is drained before reuse)
            if (dact) begin
                dc <= dc + 1'b1;
                if (dc == 4'd15) begin dact <= 0; if (d_se) ready[d_ob] <= 1; end
            end
            if (v2 && l2) begin dact <= 1; dc <= 0; d_ct <= ct2; d_ob <= ob2; d_se <= se2; end
            if (released) ready[~wr_wb] <= 0;

            // ---- request sequencing
            if (st != IDLE && st != STOP && st != SCRUB && (abort || fault)) begin
                st <= STOP; run <= 0;
            end else case (st)
                IDLE: if (go) begin
                    rows_left <= M; xblk <= xa; yblk <= ya; ready <= 0; ob <= 0;
                    st <= LDX; rd_start <= 1; ld_w <= 0; rd_addr_c <= xa;
                    rd_nb_c <= (M > TM ? TM : M) * K[9:5];
                    lkw <= 0; lb <= 0; lbase <= 0;
                end
                LDX: if (!kick && !rd_start && !rd_busy) begin
                    st <= LDW; rd_start <= 1; ld_w <= 1; ld_half <= 0; rd_addr_c <= wa;
                    rd_nb_c <= K; lkw <= 0; lb <= 0; lbase <= 0; p <= 0; wpan <= wa; ypan <= yblk;
                end
                LDW: if (!kick && !rd_start && !rd_busy) begin
                    st <= PANEL; wsel <= 0;
                    run <= 1; rt <= 0; ct <= 0; kw <= 0; xrt <= 0; wct <= 0;
                    yrt <= ypan; rows_rt <= rows_b;
                    if (!last_panel) begin
                        rd_start <= 1; ld_half <= 1; rd_addr_c <= wa + (K32 << 6);                  // panel 1
                        lkw <= 0; lb <= 0; lbase <= 0;
                    end
                end
                PANEL: if (!kick && !rd_start && !rd_busy && !comp_busy) begin
                    if (!last_panel) begin                       // next panel is already loaded
                        p <= p + 1'b1; wsel <= ~wsel; ypan <= ypan + 32'd128;
                        wpan <= wpan + (K32 << 6);                    // TN*K*2 bytes
                        run <= 1; rt <= 0; ct <= 0; kw <= 0; xrt <= 0; wct <= 0;
                        yrt <= ypan + 32'd128; rows_rt <= rows_b;
                        if (p + 2'd2 < N[11:5]) begin
                            rd_start <= 1; ld_half <= ~ld_half; rd_addr_c <= wpan + (K32 << 7);  // panel p+2
                            lkw <= 0; lb <= 0; lbase <= 0;
                        end
                    end else if (rows_left > TM) begin           // next X block
                        rows_left <= rows_left - TM; xblk <= xblk + (K32 << 8);   // TM*K*2
                        yblk <= yblk + (NB4 << 7);                                // TM*N*4
                        st <= LDX; rd_start <= 1; ld_w <= 0; rd_addr_c <= xblk + (K32 << 8);
                        rd_nb_c <= (rows_left - TM > TM ? TM : rows_left - TM) * K[9:5];
                        lkw <= 0; lb <= 0; lbase <= 0;
                    end else st <= FLUSH;
                end
                FLUSH: if (ready == 0 && !wr_busy) begin fence_req <= 1; st <= FENCE; end
                FENCE: if (fence_done) begin ok <= 1; st <= SCRUB; sc <= 0; end
                STOP: if (!rd_busy && !wr_busy && !comp_busy) begin ok <= 0; st <= SCRUB; sc <= 0; end
                SCRUB: begin
                    sc <= sc + 1'b1; ready <= 0;
                    if (sc == XD - 1) begin st <= IDLE; finished <= 1; end
                end
                default: st <= IDLE;
            endcase
        end
    end
endmodule
