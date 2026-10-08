// Simple dual-port RAM, registered read (infers M10K on Cyclone V).
// Read enable gates the output register: no toggling when the reader is idle (power).
module lmm_ram #(parameter W = 64, D = 1024, AW = $clog2(D)) (
    input              clk,
    input              we,
    input  [AW-1:0]    wa,
    input  [W-1:0]     wd,
    input              re,
    input  [AW-1:0]    ra,
    output reg [W-1:0] q
);
    reg [W-1:0] mem [0:D-1];
    always @(posedge clk) begin
        if (we) mem[wa] <= wd;
        if (re) q <= mem[ra];
    end
endmodule
