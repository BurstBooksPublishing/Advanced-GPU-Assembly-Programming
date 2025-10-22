module mem_ctrl #(
  parameter ADDR_W = 32,
  parameter DATA_W = 128,
  parameter QSZ    = 16,
  parameter ROW_W  = 16
)(
  input  wire                 clk,
  input  wire                 rst_n,
  // Read request input (single-beat handshake)
  input  wire                 r_req_v,
  input  wire [ADDR_W-1:0]    r_req_addr,
  output reg                  r_req_ready,
  // Write request input
  input  wire                 w_req_v,
  input  wire [ADDR_W-1:0]    w_req_addr,
  input  wire [DATA_W-1:0]    w_req_data,
  output reg                  w_req_ready,
  // Simple DRAM interface (abstracted single channel)
  output reg                  dram_cmd_v,
  output reg [ADDR_W-1:0]     dram_cmd_addr,
  output reg                  dram_cmd_is_write,
  input  wire                 dram_resp_v,
  input  wire [DATA_W-1:0]    dram_resp_data
);

  // FIFOs for read and write (simple circular buffer)
  reg [ADDR_W-1:0] r_q_addr [0:QSZ-1];
  reg [ADDR_W-1:0] w_q_addr [0:QSZ-1];
  reg [DATA_W-1:0] w_q_data [0:QSZ-1];
  reg [$clog2(QSZ):0] r_head, r_tail, r_cnt;
  reg [$clog2(QSZ):0] w_head, w_tail, w_cnt;

  // Row buffer tracking (simple per-channel open row)
  reg [ROW_W-1:0] open_row;
  reg             open_valid;

  // Write drain threshold
  localparam WRITE_DRAIN_TH = QSZ-4;
  reg write_drain;

  integer i;
  // enqueue/dequeue logic
  always @(posedge clk) begin
    if (!rst_n) begin
      r_head <= 0; r_tail <= 0; r_cnt <= 0;
      w_head <= 0; w_tail <= 0; w_cnt <= 0;
      r_req_ready <= 1; w_req_ready <= 1;
      dram_cmd_v <= 0; dram_cmd_addr <= 0; dram_cmd_is_write <= 0;
      open_row <= 0; open_valid <= 0; write_drain <= 0;
    end else begin
      // accept read
      if (r_req_v && r_req_ready) begin
        r_q_addr[r_tail] <= r_req_addr;
        r_tail <= r_tail + 1;
        r_cnt <= r_cnt + 1;
      end
      // accept write
      if (w_req_v && w_req_ready) begin
        w_q_addr[w_tail] <= w_req_addr;
        w_q_data[w_tail] <= w_req_data;
        w_tail <= w_tail + 1;
        w_cnt <= w_cnt + 1;
      end
      // backpressure when full
      r_req_ready <= (r_cnt < QSZ-1);
      w_req_ready <= (w_cnt < QSZ-1);

      // write drain decision
      if (w_cnt >= WRITE_DRAIN_TH) write_drain <= 1;
      else if (w_cnt == 0) write_drain <= 0;

      // scheduling: prefer reads that hit open row (FR-FCFS style)
      dram_cmd_v <= 0;
      if (r_cnt > 0 && !write_drain) begin
        // compute row tag (simple low bits)
        if (open_valid && r_q_addr[r_head][ROW_W-1:0] == open_row) begin
          // row-buffer hit, issue read
          dram_cmd_v <= 1;
          dram_cmd_addr <= r_q_addr[r_head];
          dram_cmd_is_write <= 0;
          r_head <= r_head + 1;
          r_cnt <= r_cnt - 1;
        end
      end
      // if no preferred read, drain writes or serve reads FIFO
      if (!dram_cmd_v) begin
        if (write_drain && w_cnt > 0) begin
          dram_cmd_v <= 1;
          dram_cmd_addr <= w_q_addr[w_head];
          dram_cmd_is_write <= 1;
          w_head <= w_head + 1;
          w_cnt <= w_cnt - 1;
        end else if (r_cnt > 0) begin
          dram_cmd_v <= 1;
          dram_cmd_addr <= r_q_addr[r_head];
          dram_cmd_is_write <= 0;
          r_head <= r_head + 1;
          r_cnt <= r_cnt - 1;
        end
      end

      // update open row on issued command (abstract)
      if (dram_cmd_v) begin
        open_row <= dram_cmd_addr[ROW_W-1:0];
        open_valid <= 1;
      end

      // handle response (just consume)
      if (dram_resp_v) begin
        // response routing and ROB retire are omitted for brevity
      end
    end
  end

endmodule