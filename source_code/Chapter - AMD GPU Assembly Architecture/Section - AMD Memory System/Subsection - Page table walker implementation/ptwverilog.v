module ptw #(
  parameter ADDR_W = 48,
  parameter DATA_W = 64,
  parameter LEVELS = 3  // configurable number of page-table levels
)(
  input  wire                 clk,
  input  wire                 rst_n,
  // New walk request from TLB subsystem
  input  wire                 req_v,               // valid new walk
  input  wire [ADDR_W-1:0]    req_va,              // virtual address
  input  wire [7:0]           req_vmid,            // VM identifier
  output reg                  req_ack,
  // Memory read (AXI-like) interface
  output reg  [ADDR_W-1:0]    mem_araddr,
  output reg                  mem_arvalid,
  input  wire                 mem_arready,
  input  wire [DATA_W-1:0]    mem_rdata,
  input  wire                 mem_rvalid,
  output reg                  mem_rready,
  // Refill / fault output
  output reg                  resp_v,
  output reg [ADDR_W-1:0]     resp_pa,
  output reg                  resp_fault,
  output reg [7:0]            resp_vmid
);

localparam IDLE = 2'd0, ISSUE = 2'd1, WAIT = 2'd2, RESP = 2'd3;

reg [1:0] state;
reg [$clog2(LEVELS+1)-1:0] level; // current level index (0..LEVELS)
reg [ADDR_W-1:0] cur_va;
reg [ADDR_W-1:0] pte_entry; // raw PTE data (assumes fits in DATA_W)
reg present_bit;

// Simple page-table level translation: compute index address for level
function [ADDR_W-1:0] level_addr;
  input [ADDR_W-1:0] va;
  input [$clog2(LEVELS+1)-1:0] lvl;
  reg [ADDR_W-1:0] base;
  begin
    // Example: derive synthetic PTE fetch address from VA and level
    base = {16'h0, lvl, 32'h0}; // placeholder base per level
    level_addr = base | (va >> (12 + lvl*9)); // index shift per-level
  end
endfunction

always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    state <= IDLE;
    req_ack <= 1'b0;
    mem_arvalid <= 1'b0;
    mem_rready <= 1'b0;
    resp_v <= 1'b0;
    level <= 0;
    cur_va <= {ADDR_W{1'b0}};
    resp_fault <= 1'b0;
    resp_pa <= {ADDR_W{1'b0}};
    resp_vmid <= 8'h0;
  end else begin
    // Default outputs
    req_ack <= 1'b0;
    resp_v <= 1'b0;

    case (state)
      IDLE: begin
        if (req_v) begin
          // accept request and start at level 0
          cur_va <= req_va;
          resp_vmid <= req_vmid;
          level <= 0;
          state <= ISSUE;
          req_ack <= 1'b1;
        end
      end

      ISSUE: begin
        // Form address for current level and issue AR
        mem_araddr  <= level_addr(cur_va, level);
        mem_arvalid <= 1'b1;
        mem_rready  <= 1'b0;
        if (mem_arvalid && mem_arready) begin
          mem_arvalid <= 1'b0;
          state <= WAIT;
        end
      end

      WAIT: begin
        if (mem_rvalid) begin
          mem_rready <= 1'b1;
          pte_entry <= mem_rdata;
          // parse present bit (LSB assumed present)
          present_bit <= mem_rdata[0];
          mem_rready <= 1'b0;
          if (!mem_rdata[0]) begin
            // not present -> fault
            resp_fault <= 1'b1;
            resp_v <= 1'b1;
            resp_pa <= {ADDR_W{1'b0}};
            state <= RESP;
          end else if (level == (LEVELS-1)) begin
            // final level: produce physical address (ex: PTE[DATA_W-1:12] << 12)
            resp_pa <= {mem_rdata[DATA_W-1:12], 12'h0};
            resp_fault <= 1'b0;
            resp_v <= 1'b1;
            state <= RESP;
          end else begin
            // advance to next level
            level <= level + 1;
            state <= ISSUE;
          end
        end
      end

      RESP: begin
        // handshake with TLB refill consumer
        // consumer must accept resp_v in same cycle; we assume single-cycle acceptance
        // clear outputs and go to IDLE
        state <= IDLE;
      end

      default: state <= IDLE;
    endcase
  end
end

endmodule