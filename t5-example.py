import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

def main():
    st.title("Text Summarization")

    summarizer = pipeline(
        task="summarization",
        model='UrukHan/t5-russian-summarization',
        min_length=20,
        max_length=40,
        truncation=True,
        model_kwargs={"cache_dir": '/Documents/Huggin_Face/'},
    ) 

    # User input
    input_text = st.text_area("Enter the text you want to summarize:", height=200)

    if st.button("Summarize"):
        if input_text:
            # Generate the summary
            output = summarizer(input_text, max_length=3000, min_length=30, do_sample=False)
            summary = output[0]['summary_text']

            # Display the summary as bullet points
            st.subheader("Summary:")
            bullet_points = summary.split(". ")
            for point in bullet_points:
                st.write(f"- {point}")
        else:
            st.warning("Please enter text to summarize.")


if __name__ == "__main__":
    main()