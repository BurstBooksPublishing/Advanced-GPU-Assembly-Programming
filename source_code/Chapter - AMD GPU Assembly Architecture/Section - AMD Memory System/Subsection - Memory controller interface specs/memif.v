module mem_ctrl_iface #(
  parameter ADDR_W = 48,       // address width
  parameter DATA_W = 256,      // data width (bits)
  parameter DEPTH  = 16        // request FIFO depth
)(
  input  wire                   clk,
  input  wire                   rst_n,
  // Request channel (ingress)
  input  wire                   req_valid,
  output reg                    req_ready,
  input  wire [ADDR_W-1:0]      req_addr,
  input  wire [7:0]             req_len,   // beats
  input  wire [1:0]             req_type,  // 00=R,01=W,10=ATOM
  input  wire [7:0]             req_qos,
  // Downstream memory bus
  output reg                    mem_valid,
  input  wire                   mem_ready,
  output reg [ADDR_W-1:0]       mem_addr,
  output reg                    mem_write,
  output reg [DATA_W-1:0]       mem_wdata,
  // Memory response (from DRAM/HBM controller)
  input  wire                   mem_resp_valid,
  input  wire [DATA_W-1:0]      mem_resp_data,
  output reg                    resp_ready,
  // Response to requester
  output reg                    resp_valid,
  output reg [DATA_W-1:0]       resp_data,
  // Status
  output reg [31:0]             outstanding
);

  // FIFO storage
  reg [ADDR_W-1:0] fifo_addr [0:DEPTH-1];
  reg [7:0]        fifo_len  [0:DEPTH-1];
  reg [1:0]        fifo_type [0:DEPTH-1];
  reg [7:0]        fifo_qos  [0:DEPTH-1];
  reg              fifo_v    [0:DEPTH-1];

  integer i;
  reg [$clog2(DEPTH):0] wr_ptr, rd_ptr, count;

  // Push request into FIFO
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_ptr <= 0; rd_ptr <= 0; count <= 0;
      outstanding <= 0;
      req_ready <= 1;
      mem_valid <= 0; mem_addr <= 0; mem_write <= 0; mem_wdata <= 0;
      resp_valid <= 0; resp_data <= 0; resp_ready <= 0;
      for (i=0;i 0)) begin
        mem_addr  <= fifo_addr[rd_ptr];
        mem_write <= (fifo_type[rd_ptr] == 2'b01);
        mem_wdata <= {DATA_W{1'b0}}; // write data path example placeholder
        mem_valid <= 1'b1;
      end else if (mem_valid && mem_ready) begin
        // commit the request to memory bus
        fifo_v[rd_ptr] <= 1'b0;
        rd_ptr <= (rd_ptr + 1) % DEPTH;
        count  <= count - 1;
        mem_valid <= 1'b0;
        outstanding <= outstanding + 1; // increment outstanding on issue
      end

      // Pass memory response to requester
      resp_ready <= 1'b1; // simple flow-control for demo
      if (mem_resp_valid && resp_ready) begin
        resp_valid <= 1'b1;
        resp_data  <= mem_resp_data;
        if (resp_valid && resp_ready) begin
          // single-beat response assumed; decrement outstanding
          if (outstanding != 0) outstanding <= outstanding - 1;
          resp_valid <= 1'b0;
        end
      end else begin
        resp_valid <= 1'b0;
      end
    end
  end

endmodule