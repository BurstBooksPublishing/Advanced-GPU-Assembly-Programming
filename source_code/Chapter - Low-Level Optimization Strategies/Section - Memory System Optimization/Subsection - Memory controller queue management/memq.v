module mem_bank_queue #(
  parameter DEPTH = 8,
  parameter ADDR_W = 32,
  parameter AGE_W  = 6
)(
  input  wire              clk,
  input  wire              rst_n,
  input  wire              enq_valid,
  input  wire [ADDR_W-1:0] enq_addr,
  input  wire              enq_is_write,
  input  wire              enq_row_hit,
  output reg               enq_ready,
  output reg               deq_valid,
  output reg [ADDR_W-1:0]  deq_addr,
  output reg               deq_is_write
);

  reg               v[DEPTH-1:0];
  reg [ADDR_W-1:0]  a[DEPTH-1:0];
  reg               w[DEPTH-1:0], hit[DEPTH-1:0];
  reg [AGE_W-1:0]   age[DEPTH-1:0];
  integer i, sel;
  reg [AGE_W-1:0] max_age;

  always @(posedge clk) begin
    if (!rst_n) begin
      for (i=0;i<DEPTH;i=i+1) v[i]<=0;
      enq_ready<=1; deq_valid<=0;
    end else begin
      // enqueue
      if (enq_valid) for (i=0;i<DEPTH;i=i+1)
        if (!v[i]) begin v[i]<=1; a[i]<=enq_addr;
          w[i]<=enq_is_write; hit[i]<=enq_row_hit; age[i]<=0; disable for;
        end
      // age
      for (i=0;i<DEPTH;i=i+1) if (v[i]) age[i]<=age[i]+1;
      // pick oldest ready row-hit
      sel=-1; max_age=0;
      for (i=0;i<DEPTH;i=i+1)
        if (v[i]&&hit[i]&&age[i]>=max_age) begin max_age=age[i]; sel=i; end
      if (sel!=-1) begin
        deq_valid<=1; deq_addr<=a[sel]; deq_is_write<=w[sel]; v[sel]<=0;
      end else deq_valid<=0;
    end
  end
endmodule