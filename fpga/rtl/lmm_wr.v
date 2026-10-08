// Avalon-MM burst write master (F2S bidirectional port) for finished output segments.
// A segment = up to 4 rows x 32 int32 (16 beats/row, 2 bursts of 8) from the ping-pong obuf.
// Every burst is checked against [lo, hi) before its first beat (second guard layer).
// `fence` issues one read of the last written address on the same port: Avalon returns it
// only after the earlier writes on this port, so DONE is never raised ahead of the data.
module lmm_wr (
    input             clk, rst,
    input             clr,                    // end of request: clear fault, segment pointer
    input      [31:0] lo, hi, row_stride,     // row_stride = N*4 bytes
    input      [1:0]  ready,                  // obuf[i] holds a finished segment
    input      [63:0] meta_addr,              // {seg1, seg0} first-row byte address
    input      [5:0]  meta_rows,              // {seg1, seg0} valid rows, 1..4
    output reg        release_seg,            // pulse: segment ~wb written (wb already flipped)
    output reg        wb,
    output            obuf_re,
    output     [6:0]  obuf_ra,
    input      [63:0] obuf_q,
    input             stop,
    input             fence_req,
    output reg        fence_done,
    output reg        fault,
    output            busy,
    output     [31:0] avm_address,
    output            avm_write,
    output            avm_read,
    output     [3:0]  avm_burstcount,
    output     [63:0] avm_writedata,
    input             avm_waitrequest,
    input             avm_readdatavalid
);
    localparam IDLE = 0, PRIME = 1, BURST = 2, FREAD = 3, FWAIT = 4;
    reg [2:0]  st;
    reg [5:0]  bi;                  // beat index inside the segment: {row[1:0], beat[3:0]}
    reg [2:0]  rows;
    reg [31:0] row_a, ba, last_a;
    wire in_range = ba >= lo && {1'b0, ba} + 33'd64 <= {1'b0, hi};
    wire take     = st == BURST && in_range && !avm_waitrequest;
    assign avm_write      = st == BURST && in_range;
    assign avm_read       = st == FREAD;
    assign avm_address    = st == FREAD ? last_a : ba;
    assign avm_burstcount = st == FREAD ? 4'd1 : 4'd8;
    assign avm_writedata  = obuf_q;
    assign obuf_re = st == PRIME || st == BURST;
    assign obuf_ra = {wb, bi + (take ? 6'd1 : 6'd0)};
    assign busy    = st != IDLE;

    always @(posedge clk) begin
        release_seg <= 0; fence_done <= 0;
        if (rst || clr) begin
            st <= IDLE; wb <= 0; fault <= 0;
        end else case (st)
            IDLE:
                if (fence_req) st <= FREAD;
                else if (ready[wb] && !stop && !fault) begin
                    bi <= 0; rows <= wb ? meta_rows[5:3] : meta_rows[2:0];
                    row_a <= wb ? meta_addr[63:32] : meta_addr[31:0];
                    ba    <= wb ? meta_addr[63:32] : meta_addr[31:0];
                    st <= PRIME;
                end
            PRIME: st <= BURST;
            BURST:
                if (!in_range) begin fault <= 1; st <= IDLE; end
                else if (take) begin
                    bi <= bi + 1'b1;
                    if (bi[2:0] == 3'd7) begin                 // burst complete
                        last_a <= ba;
                        if (bi[3]) begin                       // row complete
                            row_a <= row_a + row_stride; ba <= row_a + row_stride;
                        end else ba <= ba + 32'd64;
                        if (bi[5:4] == rows - 1'b1 && bi[3]) begin
                            release_seg <= 1; wb <= ~wb; st <= IDLE;
                        end else if (stop) st <= IDLE;
                    end
                end
            FREAD: if (!avm_waitrequest) st <= FWAIT;
            FWAIT: if (avm_readdatavalid) begin fence_done <= 1; st <= IDLE; end
            default: st <= IDLE;
        endcase
    end
endmodule
