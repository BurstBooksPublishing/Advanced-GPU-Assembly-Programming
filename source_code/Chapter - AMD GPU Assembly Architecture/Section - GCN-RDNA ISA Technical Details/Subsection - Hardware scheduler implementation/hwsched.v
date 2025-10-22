module hw_scheduler #(
  parameter W = 8,               // number of resident waves
  parameter IDW = $clog2(W)      // width of wave id
)(
  input  wire               clk,
  input  wire               rstn,
  input  wire [W-1:0]       wave_valid,   // resident descriptor valid
  input  wire [W-1:0]       wave_ready,   // ready mask (not waiting on mem)
  input  wire [W-1:0]       wave_scoreboard, // 1 = hazard present (cannot issue)
  output reg  [IDW-1:0]     issue_id,     // selected wave id (valid when issue_valid)
  output reg                issue_valid,
  input  wire               retire_ack    // ack from ALU that issued instr retired
);

  // internal round-robin pointer
  reg [IDW-1:0] rr_ptr;
  reg [W-1:0] selectable;   // candidate vector after filtering

  integer i;
  // combinational: compute selectable waves (valid, ready, not scoreboard blocked)
  always @(*) begin
    selectable = wave_valid & wave_ready & ~wave_scoreboard;
  end

  // rotate-select logic for round robin: create rotated vector and pick first set bit
  function [IDW-1:0] find_first;
    input [W-1:0] vec;
    integer j;
    begin
      find_first = {IDW{1'b0}}; // default zero
      for (j=0; j> rr_ptr) | (selectable << (W - rr_ptr));
  end

  // choose candidate from rotated vector
  reg [IDW-1:0] chosen_rot;
  reg chosen_valid;
  always @(*) begin
    if (|rot_sel) begin
      chosen_rot = find_first(rot_sel);
      chosen_valid = 1'b1;
    end else begin
      chosen_rot = {IDW{1'b0}};
      chosen_valid = 1'b0;
    end
  end

  // map rotated id back to absolute id
  wire [IDW-1:0] chosen_abs = (chosen_rot + rr_ptr) % W;

  // sequential: update rr_ptr and issue outputs
  always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
      rr_ptr <= {IDW{1'b0}};
      issue_id <= {IDW{1'b0}};
      issue_valid <= 1'b0;
    end else begin
      if (chosen_valid) begin
        issue_id <= chosen_abs;
        issue_valid <= 1'b1;
      end else begin
        issue_valid <= 1'b0;
      end
      // advance rr_ptr when an issue is accepted (retire_ack)
      if (retire_ack && chosen_valid) begin
        rr_ptr <= chosen_abs + 1; // next position after issued
      end
    end
  end

endmodule