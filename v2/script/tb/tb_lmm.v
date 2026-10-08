// Testbench for lmm_top: Avalon DDR model with random stalls/latency, bus monitor, CSR driver.
//   +test=func  : one GEMM from +mem (hex image), dump Y to +out          (driven by run_sim.py)
//   +test=sec   : negative/security suite on a small random GEMM
// Bus monitor (always on): every access must lie inside the request's X/W (reads) or Y
// (writes) region, and no access may happen while no request is running.
`timescale 1ns/1ps
module tb_lmm;
    localparam MEMW = 1 << 21;                        // 16 MiB of 64-bit words
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;                             // 100 MHz

    reg  [4:0]  csr_address = 0;
    reg         csr_read = 0, csr_write = 0;
    reg  [31:0] csr_writedata = 0;
    wire [31:0] csr_readdata;
    wire        irq;
    wire [31:0] rd_address, wr_address;
    wire        rd_read, wr_write, wr_read;
    wire [3:0]  rd_burstcount, wr_burstcount;
    wire [63:0] wr_writedata;
    reg         rd_waitrequest = 0, wr_waitrequest = 0;
    reg  [63:0] rd_readdata = 0;
    reg         rd_readdatavalid = 0, wr_readdatavalid = 0;

    lmm_top dut (.clk(clk), .rst_n(rst_n), .csr_address(csr_address), .csr_read(csr_read),
        .csr_write(csr_write), .csr_writedata(csr_writedata), .csr_readdata(csr_readdata), .irq(irq),
        .rd_address(rd_address), .rd_read(rd_read), .rd_burstcount(rd_burstcount),
        .rd_waitrequest(rd_waitrequest), .rd_readdata(rd_readdata), .rd_readdatavalid(rd_readdatavalid),
        .wr_address(wr_address), .wr_write(wr_write), .wr_read(wr_read), .wr_burstcount(wr_burstcount),
        .wr_writedata(wr_writedata), .wr_waitrequest(wr_waitrequest), .wr_readdatavalid(wr_readdatavalid));

    // ---------------- DDR model ----------------
    reg [63:0] mem [0:MEMW-1];
    integer stall_pct = 20, lat = 8;                  // +stall=<%> +lat=<cycles>
    // +bw=<beats per 100 cycles>: shared DDR budget for read + write beats (0 = unlimited);
    // at 100 MHz x 8 B, bw=100 is 800 MB/s, bw=25 is 200 MB/s.
    integer bw = 0, credit = 0;
    reg turn = 0;                                     // round-robin between ports (MPFE-like)
    reg [31:0] seed = 1;
    function [31:0] rnd; input dummy; begin seed = seed * 1103515245 + 12345; rnd = seed >> 8; end endfunction

    // read port: queue of accepted bursts, beats returned in order after `lat` cycles
    reg [31:0] rq_addr [0:1023]; reg [3:0] rq_len [0:1023]; integer rq_t [0:1023];
    integer rq_h = 0, rq_n = 0, rbeat = 0, cyc = 0;
    // write port: burst state + fence-read queue
    reg [31:0] wa_cur; integer wleft = 0, wf_t = -1; reg [31:0] wf_a;
    integer rd_beats = 0, wr_beats = 0, violations = 0, bursts_wr = 0;
    reg [31:0] lo_x, hi_x, lo_w, hi_w, lo_y, hi_y; reg active = 0;

    always @(posedge clk) begin
        cyc = cyc + 1;
        if (bw != 0 && credit < 1600) credit = credit + bw;
        rd_waitrequest <= (rnd(0) % 100) < stall_pct;
        wr_waitrequest <= (rnd(0) % 100) < stall_pct || (bw != 0 && (credit < 100 || (turn == 0 && rq_n > 0)));
        // ---- monitor
        if ((rd_read || wr_write || wr_read) && !active) begin
            violations = violations + 1; $display("VIOLATION: bus access with no running request");
        end
        if (rd_read && !((rd_address >= lo_x && rd_address + 64 <= hi_x) ||
                         (rd_address >= lo_w && rd_address + 64 <= hi_w))) begin
            violations = violations + 1; $display("VIOLATION: read %h outside X/W", rd_address);
        end
        if ((wr_write && wleft == 0 || wr_read) && !(wr_address >= lo_y && wr_address + 64 <= hi_y)) begin
            violations = violations + 1; $display("VIOLATION: write-port %h outside Y", wr_address);
        end
        // ---- read port
        if (rq_n > 1000) begin violations = violations + 1; $display("VIOLATION: model queue overflow"); end
        if (rd_read && !rd_waitrequest) begin
            rq_addr[(rq_h + rq_n) % 1024] = rd_address; rq_len[(rq_h + rq_n) % 1024] = rd_burstcount;
            rq_t[(rq_h + rq_n) % 1024] = cyc + lat; rq_n = rq_n + 1;
        end
        rd_readdatavalid <= 0;
        if (rq_n > 0 && cyc >= rq_t[rq_h] && (rnd(0) % 100) >= stall_pct / 2 &&
            (bw == 0 || (credit >= 100 && (turn == 0 || !wr_write)))) begin
            if (bw != 0) begin credit = credit - 100; turn = 1; end
            rd_readdatavalid <= 1;
            rd_readdata <= mem[rq_addr[rq_h] / 8 + rbeat];
            rd_beats = rd_beats + 1; rbeat = rbeat + 1;
            if (rbeat == rq_len[rq_h]) begin rbeat = 0; rq_h = (rq_h + 1) % 1024; rq_n = rq_n - 1; end
        end
        // ---- write port (writes complete in order; fence read returns after them)
        if (wr_write && !wr_waitrequest) begin
            if (wleft == 0) begin wa_cur = wr_address; wleft = wr_burstcount; bursts_wr = bursts_wr + 1; end
            mem[wa_cur / 8] <= wr_writedata; wa_cur = wa_cur + 8; wleft = wleft - 1; wr_beats = wr_beats + 1;
            if (bw != 0) begin credit = credit - 100; turn = 0; end
        end
        wr_readdatavalid <= 0;
        if (wr_read && !wr_waitrequest) begin wf_t = cyc + lat; wf_a = wr_address; end
        if (wf_t >= 0 && cyc >= wf_t) begin wr_readdatavalid <= 1; wf_t = -1; end
    end

    // ---------------- CSR helpers ----------------
    task csr_wr(input [4:0] a, input [31:0] d);
        begin @(negedge clk); csr_address = a; csr_writedata = d; csr_write = 1;
              @(negedge clk); csr_write = 0; end
    endtask
    task csr_rd(input [4:0] a, output [31:0] d);
        begin @(negedge clk); csr_address = a; csr_read = 1;
              @(negedge clk); csr_read = 0; d = csr_readdata; end
    endtask
    reg [31:0] r;
    task wait_idle; begin r = 32'h10; while (r[4]) csr_rd(5'h02, r); end endtask
    task setup(input [31:0] m, k, n, x, w, y, t);
        begin csr_wr(5'h05, m); csr_wr(5'h06, k); csr_wr(5'h07, n); csr_wr(5'h08, x); csr_wr(5'h09, w);
              csr_wr(5'h0A, y); csr_wr(5'h0B, t);
              lo_x = x; hi_x = x + m * k * 2; lo_w = w; hi_w = w + n * k * 2; lo_y = y; hi_y = y + m * n * 4; end
    endtask
    task run(output [7:0] code);
        begin active = 1; csr_wr(5'h01, 32'h5); wait_idle; active = 0;
              code = r[15:8]; csr_wr(5'h01, 32'h8); end           // clear done/err
    endtask

    // every RAM word and data register must be zero after a request
    integer i, j, dirty;
    task check_scrub;
        begin dirty = 0;
            for (i = 0; i < 6144; i = i + 1) begin
                if (dut.u_core.XB[0].ram.mem[i] !== 0 || dut.u_core.XB[1].ram.mem[i] !== 0 ||
                    dut.u_core.XB[2].ram.mem[i] !== 0 || dut.u_core.XB[3].ram.mem[i] !== 0) dirty = dirty + 1;
            end
            for (i = 0; i < 1536; i = i + 1)
                for (j = 0; j < 8; j = j + 1) case (j)
                    0: if (dut.u_core.WB[0].ram.mem[i] !== 0) dirty = dirty + 1;
                    1: if (dut.u_core.WB[1].ram.mem[i] !== 0) dirty = dirty + 1;
                    2: if (dut.u_core.WB[2].ram.mem[i] !== 0) dirty = dirty + 1;
                    3: if (dut.u_core.WB[3].ram.mem[i] !== 0) dirty = dirty + 1;
                    4: if (dut.u_core.WB[4].ram.mem[i] !== 0) dirty = dirty + 1;
                    5: if (dut.u_core.WB[5].ram.mem[i] !== 0) dirty = dirty + 1;
                    6: if (dut.u_core.WB[6].ram.mem[i] !== 0) dirty = dirty + 1;
                    7: if (dut.u_core.WB[7].ram.mem[i] !== 0) dirty = dirty + 1;
                endcase
            for (i = 0; i < 128; i = i + 1) if (dut.u_core.u_obuf.mem[i] !== 0) dirty = dirty + 1;
            if (dut.u_core.drain !== 0) dirty = dirty + 1;
            // RAM output registers and PE registers (v2)
            if (dut.u_core.XB[0].ram.q !== 0 || dut.u_core.XB[1].ram.q !== 0 ||
                dut.u_core.XB[2].ram.q !== 0 || dut.u_core.XB[3].ram.q !== 0) dirty = dirty + 1;
            if (dut.u_core.WB[0].ram.q !== 0 || dut.u_core.WB[1].ram.q !== 0 || dut.u_core.WB[2].ram.q !== 0 ||
                dut.u_core.WB[3].ram.q !== 0 || dut.u_core.WB[4].ram.q !== 0 || dut.u_core.WB[5].ram.q !== 0 ||
                dut.u_core.WB[6].ram.q !== 0 || dut.u_core.WB[7].ram.q !== 0) dirty = dirty + 1;
            if (dut.u_core.u_obuf.q !== 0) dirty = dirty + 1;
            for (i = 0; i < 32; i = i + 1)
                if (dut.u_core.acc[i] !== 0 || dut.u_core.ph0[i] !== 0 || dut.u_core.ph1[i] !== 0) dirty = dirty + 1;
        end
    endtask

    // ---------------- tests ----------------
    reg [8*16-1:0] test; reg [8*256-1:0] memf, outf;
    integer M, K, N, X, W, Y, T, fails = 0, t0;
    reg [7:0] code;
    task check_code(input [7:0] want, input [8*40-1:0] name);
        begin if (code !== want) begin fails = fails + 1; $display("FAIL %0s: code %0d want %0d", name, code, want); end
              else $display("PASS %0s (code %0d)", name, code); end
    endtask

    initial begin
        if (!$value$plusargs("test=%s", test)) test = "func";
        if ($value$plusargs("seed=%d", seed)) ;
        if ($value$plusargs("stall=%d", stall_pct)) ;
        if ($value$plusargs("lat=%d", lat)) ;
        if ($value$plusargs("bw=%d", bw)) ;
        if ($test$plusargs("vcd")) begin $dumpfile("lmm.vcd"); $dumpvars(0, tb_lmm); end
        for (i = 0; i < MEMW; i = i + 1) mem[i] = 64'hDEAD_BEEF_DEAD_BEEF;
        repeat (4) @(negedge clk); rst_n = 1; repeat (4) @(negedge clk);

        if (test == "func") begin
            void'($value$plusargs("mem=%s", memf)); void'($value$plusargs("out=%s", outf));
            void'($value$plusargs("M=%d", M)); void'($value$plusargs("K=%d", K)); void'($value$plusargs("N=%d", N));
            void'($value$plusargs("X=%d", X)); void'($value$plusargs("W=%d", W)); void'($value$plusargs("Y=%d", Y));
            $readmemh(memf, mem, 0, Y / 8 - 1);
            csr_wr(5'h0C, 0); csr_wr(5'h0D, MEMW * 8); csr_wr(5'h0E, 1);
            setup(M, K, N, X, W, Y, 32'hFFFF_FFFF); csr_wr(5'h03, 32'h1234);
            t0 = cyc; run(code);
            csr_rd(5'h04, r);
            if (code != 0 || r != 32'h1234) fails = fails + 1;
            $display("RESULT code=%0d done_id=%h cycles=%0d", code, r, cyc - t0);
            csr_rd(5'h10, r); $display("PERF busy_cycles=%0d", r);
            csr_rd(5'h11, r); $display("PERF mac_cycles=%0d", r);
            csr_rd(5'h12, r); $display("PERF rd_beats=%0d", r);
            csr_rd(5'h13, r); $display("PERF wr_beats=%0d", r);
            check_scrub; if (dirty) begin fails = fails + 1; $display("FAIL scrub: %0d dirty words", dirty); end
            $writememh(outf, mem, Y / 8, (Y + M * N * 4) / 8 - 1);
        end else begin
            // small random GEMM: M=9 (tail rows), K=64 (minimum, 16-cycle tiles), N=64 (2 panels)
            M = 9; K = 64; N = 64; X = 32'h1000; W = 32'h4000; Y = 32'h10000;
            for (i = 0; i < 4096; i = i + 1) mem[(X / 8) + i] = {rnd(0), rnd(0)};

            setup(M, K, N, X, W, Y, 100000); run(code); check_code(6, "start before window lock");

            csr_wr(5'h0C, 32'h1000); csr_wr(5'h0D, 32'h20000); csr_wr(5'h0E, 1);
            csr_wr(5'h0D, 32'hFFFF_FFFF); csr_rd(5'h0D, r);
            code = r == 32'h20000 ? 0 : 1; check_code(0, "window frozen after lock");

            setup(0, K, N, X, W, Y, 100000);      run(code); check_code(2, "M = 0");
            setup(1025, K, N, X, W, Y, 100000);   run(code); check_code(2, "M > 1024");
            setup(M, 32, N, X, W, Y, 100000);     run(code); check_code(2, "K < 64");
            setup(M, 800, N, X, W, Y, 100000);    run(code); check_code(2, "K > 768");
            setup(M, 100, N, X, W, Y, 100000);    run(code); check_code(2, "K % 32 != 0");
            setup(M, K, 48, X, W, Y, 100000);     run(code); check_code(2, "N % 32 != 0");
            setup(M, K, 2336, X, W, Y, 100000);   run(code); check_code(2, "N > 2304");
            setup(M, K, N, X, W, Y, 0);           run(code); check_code(2, "timeout = 0");
            setup(M, K, N, X + 8, W, Y, 100000);  run(code); check_code(3, "X misaligned");
            setup(M, K, N, 32'h0, W, Y, 100000);  run(code); check_code(4, "X below window");
            setup(M, K, N, X, W, 32'h1FFC0, 100000); run(code); check_code(4, "Y end beyond window");
            setup(M, K, N, X, W, 32'hFFFF_FFC0, 100000); run(code); check_code(4, "Y near 4 GB (wrap)");
            setup(M, K, N, X, W, X + 64, 100000); run(code); check_code(5, "Y overlaps X");
            setup(M, K, N, X, W, W, 100000);      run(code); check_code(5, "Y overlaps W");
            if (violations != 0 || rd_beats != 0 || wr_beats != 0) begin
                fails = fails + 1; $display("FAIL: rejected requests touched the bus");
            end else $display("PASS rejected requests: 0 bus beats");

            // valid run, then busy protections
            setup(M, K, N, X, W, Y, 100000); csr_wr(5'h03, 32'hA1);
            active = 1; csr_wr(5'h01, 32'h5);
            repeat (50) @(negedge clk);
            csr_wr(5'h05, 1); csr_wr(5'h01, 32'h5);              // argument write + START while busy
            wait_idle; active = 0; code = r[15:8];
            check_code(0, "valid request completes");
            csr_rd(5'h05, r); code = r == M ? 0 : 1; check_code(0, "M frozen while busy");
            csr_rd(5'h02, r); code = r[7] ? 0 : 1; check_code(0, "busy writes flagged");
            csr_rd(5'h04, r); code = r == 32'hA1 ? 0 : 1; check_code(0, "DONE_ID echoes REQ_ID");
            code = irq ? 0 : 1; check_code(0, "irq raised on done");
            csr_wr(5'h01, 32'h8);
            check_scrub; code = dirty ? 1 : 0; check_code(0, "buffers scrubbed after done");

            setup(M, K, N, X, W, Y, 300); csr_wr(5'h03, 32'hB2); run(code); check_code(7, "watchdog timeout");
            csr_rd(5'h04, r); code = r == 32'hA1 ? 0 : 1; check_code(0, "no completion id for failed request");
            check_scrub; code = dirty ? 1 : 0; check_code(0, "buffers scrubbed after timeout");

            setup(M, K, N, X, W, Y, 100000); active = 1; csr_wr(5'h01, 32'h5);
            repeat (400) @(negedge clk); csr_wr(5'h01, 32'h2);  // ABORT mid-run
            wait_idle; active = 0; code = r[15:8]; csr_wr(5'h01, 32'h8); check_code(8, "host abort");
            check_scrub; code = dirty ? 1 : 0; check_code(0, "buffers scrubbed after abort");

            // guard layer 2: corrupt the writer's bound after validation (models a scheduler bug)
            setup(M, K, N, X, W, Y, 100000); active = 1; csr_wr(5'h01, 32'h5);
            force dut.u_core.y_hi = Y;
            wait_idle; active = 0; code = r[15:8]; csr_wr(5'h01, 32'h8); release dut.u_core.y_hi;
            check_code(9, "guard blocks out-of-region write");

            code = (wr_beats % 8 == 0) ? 0 : 1; check_code(0, "no truncated write burst");
            setup(M, K, N, X, W, Y, 100000); run(code); check_code(0, "recovers after errors");
            code = violations == 0 ? 0 : 1; check_code(0, "bus monitor: 0 violations");
        end
        $display("%0s %0s violations=%0d", fails || violations ? "TEST FAILED" : "TEST PASSED", test, violations);
        $finish;
    end
endmodule
