import streamlit as st
from post_generator import PostGenerator
from models import PostPromptInput

def main():
    st.title("AI Post Generator")
    st.write("Generate social media posts that match your style and content preferences.")

    # Input section
    st.header("Input")
    prompt = st.text_area("What would you like to write about?", 
                         placeholder="Enter your post topic or idea...")
    platform = st.selectbox("Select Platform", ["twitter", "linkedin"])

    if st.button("Generate Post"):
        if not prompt:
            st.error("Please enter a prompt first!")
            return

        try:
            # Initialize generator
            with st.spinner("Generating post..."):
                generator = PostGenerator()
                
                # Create input data
                input_data = PostPromptInput(
                    prompt=prompt,
                    platform=platform
                )
                
                # Generate post
                result = generator.generate_post(input_data)

                # Display results
                st.header("Generated Post")
                st.write(result.generated_post.post)

                # Display context
                st.header("Context Used")
                for i, ctx in enumerate(result.generated_post.retrieved_contexts, 1):
                    with st.expander(f"Example {i} (Similarity: {ctx.similarity_score:.2f})"):
                        st.write(f"**Content:** {ctx.content}")
                        st.write(f"**Platform:** {ctx.platform}")
                        st.write(f"**Date:** {ctx.date}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 