module warp_scheduler #(
  parameter WARPS = 8,
  parameter IDW = 3
)(
  input  wire                 clk,
  input  wire                 rst_n,
  input  wire [WARPS-1:0]     ready_mask, // 1 = warp ready
  input  wire                 issue_accept, // downstream accepted issue
  output reg  [WARPS-1:0]     issue_onehot,
  output reg                  issue_valid
);

  reg [IDW-1:0] rr_ptr; // round-robin pointer
  integer i;
  // Compute rotated ready vector to implement round-robin fairness.
  wire [WARPS-1:0] rotated_ready;
  assign rotated_ready = (ready_mask << rr_ptr) | (ready_mask >> (WARPS - rr_ptr));

  // Priority encode rotated_ready to get first (oldest) ready warp.
  reg [IDW-1:0] sel_idx;
  reg found;
  always @(*) begin
    sel_idx = 0;
    found = 0;
    for (i = 0; i < WARPS; i = i + 1) begin
      if (!found && rotated_ready[i]) begin
        sel_idx = i[IDW-1:0];
        found = 1;
      end
    end
  end

  // Convert selection back to original index space and produce one-hot.
  reg [IDW-1:0] orig_idx;
  always @(*) begin
    if (found) begin
      orig_idx = (sel_idx + (WARPS - rr_ptr)) % WARPS;
      issue_onehot = {WARPS{1'b0}} | (1 << orig_idx);
      issue_valid = 1'b1;
    end else begin
      issue_onehot = {WARPS{1'b0}};
      issue_valid = 1'b0;
    end
  end

  // Update rr_ptr when issue accepted to rotate turn.
  always @(posedge clk or negedge rst_n) begin
    if (!rst_n) rr_ptr <= 0;
    else if (issue_valid && issue_accept) rr_ptr <= (rr_ptr + 1) % WARPS;
  end

endmodule