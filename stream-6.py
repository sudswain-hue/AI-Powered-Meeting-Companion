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
    st.title("MeetScribe")
    
    # Top section: Upload and transcript in two columns
    top_left, top_right = st.columns([1, 2])
    
    with top_left:
        st.markdown("### Upload Audio")
        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])
        
        if uploaded_file is not None:
            # Save the file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Display audio player
            st.audio(uploaded_file)
    
    # Initialize variables
    transcription_text = ""
    
    # Display transcript in right column (always present)
    with top_right:
        st.markdown("### Transcription")
        transcript_area = st.empty()
        transcript_box = transcript_area.text_area("", transcription_text, height=300, 
                                                  placeholder="Transcript will appear here...")
        
        # Download button container (initially empty)
        download_btn_container = st.empty()
    
    # Analysis results container
    analysis_container = st.container()
    
    # Processing logic
    if uploaded_file is not None:
        # Processing indicator
        progress_placeholder = st.empty()
        progress_placeholder.info("Transcribing audio...")
        
        # Process the audio file
        transcription_text = transcript(tmp_path)
        progress_placeholder.success("Transcription complete!")
        
        # Update transcript area
        transcript_box = transcript_area.text_area("", transcription_text, height=300)
        
        # Add download button for transcript
        download_btn_container.download_button(
            label="Download Transcript",
            data=transcription_text,
            file_name="transcript.txt",
            mime="text/plain"
        )
        
        # Analysis section
        with analysis_container:
            st.markdown("---")
            st.markdown("### Analysis Results")
            
            # Create tabs for the analysis sections
            tab1, tab2, tab3 = st.tabs(["Summary", "Action Items", "Sentiment Analysis"])
            
            # Summary Tab
            with tab1:
                with st.spinner("Generating summary..."):
                    summary_text = summary(transcription_text)
                    st.text_area("Meeting Summary", summary_text, height=300)
                    st.download_button(
                        label="Download Summary",
                        data=summary_text,
                        file_name="summary.txt",
                        mime="text/plain"
                    )
            
            # Action Items Tab
            with tab2:
                with st.spinner("Extracting action items..."):
                    actions = action_items(transcription_text)
                    if actions:
                        action_text = ""
                        
                        for idx, item in enumerate(actions, 1):
                            st.markdown(f"**{idx}. Task:** {item['task']}")
                            st.markdown(f"   **Assigned To:** {', '.join(item['owners']) if item['owners'] else 'N/A'}")
                            st.markdown(f"   **Deadlines:** {', '.join(item['deadlines']) if item['deadlines'] else 'N/A'}")
                            st.markdown("   ---")
                            
                            # Build text for download
                            action_text += f"{idx}. Task: {item['task']}\n"
                            action_text += f"   Assigned To: {', '.join(item['owners']) if item['owners'] else 'N/A'}\n"
                            action_text += f"   Deadlines: {', '.join(item['deadlines']) if item['deadlines'] else 'N/A'}\n\n"
                        
                        if action_text:
                            st.download_button(
                                label="Download Action Items",
                                data=action_text,
                                file_name="action_items.txt",
                                mime="text/plain"
                            )
                    else:
                        st.info("No action items found in this meeting recording.")
                        st.markdown("""
                        Action items are typically introduced with words like:
                        - please
                        - send
                        - review
                        - schedule
                        - complete
                        - update
                        - assign
                        - remind
                        """)
            
            # Sentiment Analysis Tab
            with tab3:
                with st.spinner("Performing sentiment analysis..."):
                    sentiments = sentiment_analysis(transcription_text)
                    
                    # Create metrics for sentiment stats
                    pos_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
                    neu_count = sum(1 for s in sentiments if s['sentiment'] == 'neutral')
                    neg_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
                    total = len(sentiments) if sentiments else 1  # Avoid division by zero
                    
                    # Display sentiment distribution
                    st.markdown("#### Sentiment Distribution")
                    stat_cols = st.columns(3)
                    with stat_cols[0]:
                        st.metric("Positive", f"{pos_count} ({pos_count/total*100:.1f}%)")
                    with stat_cols[1]:
                        st.metric("Neutral", f"{neu_count} ({neu_count/total*100:.1f}%)")
                    with stat_cols[2]:
                        st.metric("Negative", f"{neg_count} ({neg_count/total*100:.1f}%)")
                    
                    # Detailed sentiment results
                    st.markdown("#### Detailed Sentiment Analysis")
                    sentiment_text = ""
                    
                    # Create a container with fixed height for detailed results
                    detailed_sentiment = st.container()
                    with detailed_sentiment:
                        # Create an expander for each sentiment category
                        if pos_count > 0:
                            with st.expander(f"Positive Statements ({pos_count})"):
                                for idx, result in enumerate(sentiments, 1):
                                    if result['sentiment'] == 'positive':
                                        st.markdown(f"**{idx}. Text:** {result['text']}")
                                        st.markdown(f"   **Score:** {result['score']:.2f}")
                                        st.markdown("   ---")
                        
                        if neu_count > 0:
                            with st.expander(f"Neutral Statements ({neu_count})"):
                                for idx, result in enumerate(sentiments, 1):
                                    if result['sentiment'] == 'neutral':
                                        st.markdown(f"**{idx}. Text:** {result['text']}")
                                        st.markdown(f"   **Score:** {result['score']:.2f}")
                                        st.markdown("   ---")
                        
                        if neg_count > 0:
                            with st.expander(f"Negative Statements ({neg_count})"):
                                for idx, result in enumerate(sentiments, 1):
                                    if result['sentiment'] == 'negative':
                                        st.markdown(f"**{idx}. Text:** {result['text']}")
                                        st.markdown(f"   **Score:** {result['score']:.2f}")
                                        st.markdown("   ---")
                    
                    # Build text for download
                    for idx, result in enumerate(sentiments, 1):
                        sentiment_text += f"{idx}. Text: {result['text']}\n"
                        sentiment_text += f"   Sentiment: {result['sentiment']} (Score: {result['score']:.2f})\n\n"
                    
                    if sentiment_text:
                        st.download_button(
                            label="Download Sentiment Analysis",
                            data=sentiment_text,
                            file_name="sentiment_analysis.txt",
                            mime="text/plain"
                        )
    
    # Add signature at the bottom right
    st.markdown("---")
    signature_col1, signature_col2 = st.columns([3, 1])
    with signature_col2:
        st.markdown("<div style='text-align: right; font-style: italic; color: #666;'>Developed by CodeBrew</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()