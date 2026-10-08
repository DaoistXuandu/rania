// Avalon-MM burst read master (F2S read-only port). Reads `nburst` bursts of 8 x 64-bit beats
// starting at `addr`, contiguous. Every burst is checked against [lo, hi) before it is issued
// (second guard layer: the validator already proved this; a failure here means a scheduler bug).
module lmm_rd (
    input             clk, rst,
    input             clr,             // end of request: clear fault
    input             start,           // pulse; addr/nburst/lo/hi sampled
    input      [31:0] addr, lo, hi,
    input      [15:0] nburst,
    input             stop,            // abort: issue nothing new, still accept outstanding beats
    output            busy,
    output reg        fault,
    output     [31:0] avm_address,
    output            avm_read,
    output     [3:0]  avm_burstcount,
    input             avm_waitrequest,
    input      [63:0] avm_readdata,
    input             avm_readdatavalid,
    output            beat_valid,
    output     [63:0] beat_data
);
    reg [31:0] a;
    reg [31:0] lim_lo, lim_hi;
    reg [15:0] left;                   // bursts not yet issued
    reg [11:0] pend;                   // beats issued but not yet returned (capped, cannot wrap)
    wire in_range = a >= lim_lo && {1'b0, a} + 33'd64 <= {1'b0, lim_hi};
    wire want     = left != 0 && !stop && !fault && pend < 12'd256;   // <= 32 bursts in flight
    assign avm_read       = want && in_range;
    assign avm_address    = a;
    assign avm_burstcount = 4'd8;
    wire issued = avm_read && !avm_waitrequest;
    assign busy       = want || pend != 0;
    assign beat_valid = avm_readdatavalid;
    assign beat_data  = avm_readdata;

    always @(posedge clk) begin
        if (rst) begin
            left <= 0; pend <= 0; fault <= 0;
        end else if (clr) begin
            left <= 0; fault <= 0;
        end else begin
            if (start) begin
                a <= addr; left <= nburst; lim_lo <= lo; lim_hi <= hi;
            end else begin
                if (want && !in_range) begin fault <= 1; left <= 0; end
                if (issued) begin a <= a + 32'd64; left <= left - 1'b1; end
                if (stop) left <= 0;
            end
            pend <= pend + (issued ? 12'd8 : 12'd0) - (avm_readdatavalid ? 12'd1 : 12'd0);
        end
    end
endmodule
