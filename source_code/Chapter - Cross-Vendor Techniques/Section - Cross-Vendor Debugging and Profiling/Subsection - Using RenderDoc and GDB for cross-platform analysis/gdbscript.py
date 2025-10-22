# gdb_inspect.py -- run with: gdb -q -x gdb_inspect.py --args ./my_app
import gdb, struct

# Break on Vulkan submit; adjust to your loader symbol if needed
class VkSubmitBreakpoint(gdb.Breakpoint):
    def stop(self):
        # Example: read first argument (pointer to VkQueue) and second (submit count)
        try:
            # On x86_64, first arg in rdi, second in rsi for SysV ABI
            regs = gdb.selected_frame().read_register
            submit_ptr = int(regs("rsi"))  # adjust if symbol differs
            print(f"vkQueueSubmit called; submit_struct_ptr=0x{submit_ptr:x}")
            # Read memory: print first 64 bytes at pointer
            mem = gdb.selected_inferior().read_memory(submit_ptr, 64)
            hexdata = ' '.join(f"{b:02x}" for b in mem)
            print("submit struct (first 64 bytes):", hexdata)
        except Exception as e:
            print("inspect error:", e)
        return False  # continue execution

# Set breakpoint at loader symbol; change to \lstinline|vkQueueSubmit| if available
VkSubmitBreakpoint("vkQueueSubmit")
print("Breakpoint set on vkQueueSubmit; run to capture pre-submit state.")
# Optional: finish script after run; user can interact after break.