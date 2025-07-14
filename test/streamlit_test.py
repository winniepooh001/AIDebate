
import streamlit as st
import asyncio
import time

# Initialize session state
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False
if 'last_auto_time' not in st.session_state:
    st.session_state.last_auto_time = 0

async def do_async_work():
    """Simulate async work like LLM call"""
    await asyncio.sleep(1)  # Simulate processing time

    # Update counter and add message
    st.session_state.counter += 1
    current_time = time.strftime('%H:%M:%S')
    st.session_state.messages.append(f"Step {st.session_state.counter} completed at {current_time}")

    # Stop after 5 steps
    if st.session_state.counter >= 5:
        st.session_state.is_running = False
        st.session_state.auto_mode = False
        return False
    return True

def main():
    st.title("🧪 Fixed Auto-Timer Test")
    st.markdown("Testing real-time UI updates with proper auto-advance")

    # Show current state
    st.metric("Counter", st.session_state.counter)
    st.write(f"Running: {st.session_state.is_running}")
    st.write(f"Auto Mode: {st.session_state.auto_mode}")

    # Show time until next auto step
    if st.session_state.auto_mode and st.session_state.is_running:
        time_since_last = time.time() - st.session_state.last_auto_time
        time_remaining = max(0, 3 - time_since_last)
        st.write(f"Next auto step in: {time_remaining:.1f} seconds")

    # Control buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 Start"):
            st.session_state.is_running = True
            st.session_state.counter = 0
            st.session_state.messages = []
            st.session_state.auto_mode = False
            st.success("Started!")
            st.rerun()

    with col2:
        if st.button("➡️ Next Step", disabled=not st.session_state.is_running):
            with st.spinner("Processing..."):
                result = asyncio.run(do_async_work())
            st.rerun()

    with col3:
        if st.button("🔄 Auto Mode", disabled=not st.session_state.is_running):
            st.session_state.auto_mode = True
            st.session_state.last_auto_time = time.time()
            st.info("Auto mode enabled! Will step every 3 seconds.")
            st.rerun()

    with col4:
        if st.button("🛑 Reset"):
            st.session_state.counter = 0
            st.session_state.messages = []
            st.session_state.is_running = False
            st.session_state.auto_mode = False
            st.rerun()

    # Show messages
    st.subheader("Messages")
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            st.write(f"{ i +1}. {msg}")
    else:
        st.info("No messages yet")

    # Completion check
    if st.session_state.counter >= 5:
        st.success("🎉 Process completed!")
        if st.button("🎉 Celebrate"):
            st.balloons()

    # Auto-advance with proper timing (at the very end)
    if st.session_state.auto_mode and st.session_state.is_running:
        current_time = time.time()
        time_since_last = current_time - st.session_state.last_auto_time

        if time_since_last >= 3:  # 3 seconds between steps
            st.session_state.last_auto_time = current_time

            with st.spinner(f"Auto-processing step {st.session_state.counter + 1}..."):
                result = asyncio.run(do_async_work())

            st.rerun()
        else:
            # Not time yet, but refresh the page to check again
            time.sleep(0.1)  # Small delay
            st.rerun()

if __name__ == "__main__":
    main()