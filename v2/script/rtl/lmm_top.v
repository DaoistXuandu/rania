// LMM accelerator top: control/status registers, request validation, watchdog, IRQ.
//   CSR: Avalon-MM slave, 32-bit, read latency 1 (behind the lightweight HPS-to-FPGA bridge)
//   rd_*: Avalon-MM read master  -> F2S SDRAM port 0 (read-only), X and W
//   wr_*: Avalon-MM master       -> F2S SDRAM port 1 (writes Y, one fence read)
// Security invariants (see tb/tb_lmm.v):
//   * no bus transaction unless the request passed every check below (fail closed);
//   * window registers are write-once until reset; argument registers are frozen while busy,
//     so what was validated is exactly what runs (no time-of-check/time-of-use gap);
//   * every buffer is zeroed after each request, success or not.
module lmm_top (
    input             clk, rst_n,
    input      [4:0]  csr_address,          // word address
    input             csr_read, csr_write,
    input      [31:0] csr_writedata,
    output reg [31:0] csr_readdata,
    output            irq,
    output     [31:0] rd_address,
    output            rd_read,
    output     [3:0]  rd_burstcount,
    input             rd_waitrequest,
    input      [63:0] rd_readdata,
    input             rd_readdatavalid,
    output     [31:0] wr_address,
    output            wr_write, wr_read,
    output     [3:0]  wr_burstcount,
    output     [63:0] wr_writedata,
    input             wr_waitrequest,
    input             wr_readdatavalid
);
    localparam MMAX = 1024, KMAX = 768, NMAX = 2304;
    localparam E_ARG = 2, E_ALIGN = 3, E_WINDOW = 4, E_OVERLAP = 5, E_UNLOCKED = 6,
               E_TIMEOUT = 7, E_ABORT = 8, E_GUARD = 9;
    localparam IDLE = 0, CHECK = 1, RUN = 2;

    // reset: asynchronous assert, synchronous release
    reg [1:0] rs;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rs <= 2'b00;
        else        rs <= {rs[0], 1'b1};
    wire rst = !rs[1];

    reg [1:0]  st;
    reg [31:0] M, K, N, X, W, Y, TIMEOUT, REQ_ID, DONE_ID, WIN_LO, WIN_HI;
    reg        win_lock, irq_en, done, err, rejected, abort_req;
    reg [7:0]  err_code;
    reg [31:0] wd_cnt, pf_cycles, pf_mac, pf_rd, pf_wr;
    wire       busy = st != IDLE;

    // ---------------- validation (combinational on frozen registers) ----------------
    // Sizes are only used once bad_arg is false, so 11/10/12-bit operands are enough
    // (max 9.4 MB); ends are computed in 34 bits so a base near 4 GB cannot wrap.
    wire [33:0] xs = M[10:0] * K[9:0] * 2, ws = N[11:0] * K[9:0] * 2, ys = M[10:0] * N[11:0] * 4;
    wire [33:0] xe = X + xs, we_ = W + ws, ye = Y + ys;
    wire bad_arg = M == 0 || M > MMAX || K < 64 || K > KMAX || K[4:0] != 0 ||
                   N == 0 || N > NMAX || N[4:0] != 0 || TIMEOUT == 0;
    wire bad_align  = X[5:0] != 0 || W[5:0] != 0 || Y[5:0] != 0;
    wire bad_window = X < WIN_LO || W < WIN_LO || Y < WIN_LO ||
                      xe > WIN_HI || we_ > WIN_HI || ye > WIN_HI;
    wire bad_overlap = (Y < xe && X < ye) || (Y < we_ && W < ye);
    wire [7:0] check = !win_lock ? E_UNLOCKED : bad_arg ? E_ARG : bad_align ? E_ALIGN :
                       bad_window ? E_WINDOW : bad_overlap ? E_OVERLAP : 8'd0;

    // ---------------- core ----------------
    reg  go;
    wire finished, ok, fault, core_busy, mac_active;
    lmm_core u_core (.clk(clk), .rst(rst), .go(go), .abort(abort_req || st == RUN && wd_cnt >= TIMEOUT),
        .M(M[10:0]), .K(K[9:0]), .N(N[11:0]), .xa(X), .wa(W), .ya(Y),
        .x_lo(X), .x_hi(xe[31:0]), .w_lo(W), .w_hi(we_[31:0]), .y_lo(Y), .y_hi(ye[31:0]),
        .finished(finished), .ok(ok), .fault(fault), .busy(core_busy), .mac_active(mac_active),
        .rd_address(rd_address), .rd_read(rd_read), .rd_burstcount(rd_burstcount),
        .rd_waitrequest(rd_waitrequest), .rd_readdata(rd_readdata), .rd_readdatavalid(rd_readdatavalid),
        .wr_address(wr_address), .wr_write(wr_write), .wr_read(wr_read), .wr_burstcount(wr_burstcount),
        .wr_writedata(wr_writedata), .wr_waitrequest(wr_waitrequest), .wr_readdatavalid(wr_readdatavalid));

    assign irq = irq_en && (done || err);

    // ---------------- registers + control ----------------
    always @(posedge clk) begin
        go <= 0;
        if (rst) begin
            st <= IDLE; win_lock <= 0; irq_en <= 0; done <= 0; err <= 0; rejected <= 0;
            abort_req <= 0; err_code <= 0; WIN_LO <= 0; WIN_HI <= 0; TIMEOUT <= 0;
            M <= 0; K <= 0; N <= 0; X <= 0; W <= 0; Y <= 0; REQ_ID <= 0; DONE_ID <= 0;
            pf_cycles <= 0; pf_mac <= 0; pf_rd <= 0; pf_wr <= 0;
        end else begin
            if (csr_write) case (csr_address)
                5'h01: begin                                     // CTRL
                    irq_en <= csr_writedata[2];
                    if (csr_writedata[3]) begin done <= 0; err <= 0; err_code <= 0; rejected <= 0; end
                    if (csr_writedata[1] && busy) abort_req <= 1;
                    if (csr_writedata[0]) begin
                        if (busy) rejected <= 1;
                        else begin
                            st <= CHECK; done <= 0; err <= 0; err_code <= 0;
                            pf_cycles <= 0; pf_mac <= 0; pf_rd <= 0; pf_wr <= 0; wd_cnt <= 0;
                        end
                    end
                end
                5'h0C: if (!win_lock) WIN_LO <= csr_writedata;
                5'h0D: if (!win_lock) WIN_HI <= csr_writedata;
                5'h0E: if (csr_writedata[0]) win_lock <= 1;     // until reset / reconfiguration
                default:
                    if (busy) rejected <= 1;                    // arguments frozen while busy
                    else case (csr_address)
                        5'h03: REQ_ID <= csr_writedata;
                        5'h05: M <= csr_writedata;
                        5'h06: K <= csr_writedata;
                        5'h07: N <= csr_writedata;
                        5'h08: X <= csr_writedata;
                        5'h09: W <= csr_writedata;
                        5'h0A: Y <= csr_writedata;
                        5'h0B: TIMEOUT <= csr_writedata;
                        default: ;
                    endcase
            endcase

            case (st)
                CHECK: if (check != 0) begin                    // rejected before any bus access
                           st <= IDLE; err <= 1; err_code <= check;
                       end else begin st <= RUN; go <= 1; end
                RUN: begin
                    wd_cnt <= wd_cnt + 1'b1; pf_cycles <= pf_cycles + 1'b1;
                    pf_mac <= pf_mac + mac_active;
                    pf_rd  <= pf_rd + (rd_readdatavalid ? 1'b1 : 1'b0);
                    pf_wr  <= pf_wr + (wr_write && !wr_waitrequest ? 1'b1 : 1'b0);
                    if (wd_cnt >= TIMEOUT && err_code == 0) err_code <= E_TIMEOUT;
                    if (abort_req && err_code == 0) err_code <= E_ABORT;
                    if (fault && err_code == 0) err_code <= E_GUARD;
                    if (finished) begin
                        st <= IDLE; abort_req <= 0;
                        if (ok) begin done <= 1; DONE_ID <= REQ_ID; end
                        else err <= 1;
                    end
                end
                default: ;
            endcase
        end
    end

    always @(posedge clk)
        if (csr_read) case (csr_address)
            5'h00: csr_readdata <= 32'h4C4D4D32;                 // "LMM2"
            5'h02: csr_readdata <= {16'd0, err_code, rejected, err, done, busy, 2'd0, st};
            5'h03: csr_readdata <= REQ_ID;
            5'h04: csr_readdata <= DONE_ID;
            5'h05: csr_readdata <= M;
            5'h06: csr_readdata <= K;
            5'h07: csr_readdata <= N;
            5'h08: csr_readdata <= X;
            5'h09: csr_readdata <= W;
            5'h0A: csr_readdata <= Y;
            5'h0B: csr_readdata <= TIMEOUT;
            5'h0C: csr_readdata <= WIN_LO;
            5'h0D: csr_readdata <= WIN_HI;
            5'h0E: csr_readdata <= {31'd0, win_lock};
            5'h10: csr_readdata <= pf_cycles;
            5'h11: csr_readdata <= pf_mac;
            5'h12: csr_readdata <= pf_rd;
            5'h13: csr_readdata <= pf_wr;
            default: csr_readdata <= 0;
        endcase
endmodule
