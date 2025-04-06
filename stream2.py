import streamlit as st
import whisper
from transformers import pipeline
import spacy
import tempfile
import os

def transcript(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcription_text = result["text"]
    return transcription_text

def summary(transcription_text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_output = summarizer(transcription_text, max_length=70, min_length=20, do_sample=False)
    return summary_output[0]['summary_text']

def action_items(transcription_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(transcription_text)
    action_keywords = ["please", "send", "review", "schedule", "complete", "update", "assign", "remind"]

    def is_action_item(sentence):
        return any(keyword in sentence.text.lower() for keyword in action_keywords)

    extracted_items = []
    for sent in doc.sents:
        if is_action_item(sent):
            persons = [ent.text for ent in sent.ents if ent.label_ == "PERSON"]
            deadlines = [ent.text for ent in sent.ents if ent.label_ in ["DATE", "TIME"]]
            extracted_items.append({
                "task": sent.text.strip(),
                "owners": persons,
                "deadlines": deadlines
            })
    return extracted_items

def map_stars_to_sentiment(label):
    stars = int(label.split()[0])
    if stars <= 2:
        return "negative"
    elif stars == 3:
        return "neutral"
    else:
        return "positive"

def sentiment_analysis(transcription_text):
    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    sentences = [sentence.strip() for sentence in transcription_text.split('.') if sentence.strip()]
    sentiment_results = []
    for sentence in sentences:
        result = classifier(sentence)[0]
        sentiment = map_stars_to_sentiment(result["label"])
        sentiment_results.append({
            "text": sentence,
            "label": result["label"],
            "score": result["score"],
            "sentiment": sentiment
        })
    return sentiment_results

def main():
    st.set_page_config(layout="wide")
    st.title("Meeting Transcription and Analysis App")
    
    # Create two columns for the layout
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.subheader("Upload Audio")
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])
        
        if uploaded_file is not None:
            # Save the file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Display audio player for the uploaded file
            st.subheader("Audio File")
            st.audio(uploaded_file)
            
            # Add processing indicators
            progress_placeholder = st.empty()
            progress_placeholder.info("Transcribing audio...")
            
            # Process the audio file
            transcription_text = transcript(tmp_path)
            
            # Create tabs for analysis results
            tabs = st.tabs(["Summary", "Action Items", "Sentiment Analysis"])
            
            with tabs[0]:
                with st.spinner("Generating summary..."):
                    summary_text = summary(transcription_text)
                    st.subheader("Summary")
                    st.write(summary_text)
            
            with tabs[1]:
                with st.spinner("Extracting action items..."):
                    actions = action_items(transcription_text)
                    st.subheader("Action Items")
                    if actions:
                        for item in actions:
                            st.markdown(f"**Task:** {item['task']}")
                            st.markdown(f"**Assigned To:** {', '.join(item['owners']) if item['owners'] else 'N/A'}")
                            st.markdown(f"**Deadlines:** {', '.join(item['deadlines']) if item['deadlines'] else 'N/A'}")
                            st.markdown("---")
                    else:
                        st.write("No action items found.")
            
            with tabs[2]:
                with st.spinner("Performing sentiment analysis..."):
                    sentiments = sentiment_analysis(transcription_text)
                    st.subheader("Sentiment Analysis")
                    for result in sentiments:
                        st.markdown(f"**Text:** {result['text']}")
                        st.markdown(f"**Sentiment:** {result['sentiment']} (Label: {result['label']}, Confidence: {result['score']:.2f})")
                        st.markdown("---")
            
            # Update progress
            progress_placeholder.success("Processing complete!")
            
    with right_col:
        if uploaded_file is not None:
            st.subheader("Transcription")
            st.text_area("Full Transcript", transcription_text, height=500)
            
            # Add download button for transcript
            st.download_button(
                label="Download Transcript",
                data=transcription_text,
                file_name="transcript.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()