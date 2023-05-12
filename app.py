import re
import os
import time
import yaml
from typing import Dict, Any
import streamlit as st
from numpy import sum as npsum
import numpy as np
import pickle
import scipy
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime
from processor.transformers.chat_eda import WhatsAppProcess, sorted_authors_df,\
    statistics, process_data, WhatsAppConfig
from processor.graphs.charts import pie_display_emojis, time_series_plot,\
    most_active_member, most_active_day,\
    max_words_used, top_media_contributor, who_shared_links,\
    sentiment_analysis, most_suitable_day, most_suitable_hour,word_frequency_count,\
    user_activity_over_time,sentiment_analysis_ind


st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="Whatsgraph💬")
nav_area = st.empty()
st.title("Whatsgraph💬")
st.header("Your messages as stats")

def add_multilingual_stopwords() -> Dict:
    multilingul_list = []
    for file in os.listdir('configs/stopwords'):
        stopword = open('configs/stopwords/' + file, "r",encoding="utf-8")
        for word in stopword:
            word = re.sub('[\n]', '', word)
            multilingul_list.append(word)
    return set(STOPWORDS).union(set(multilingul_list))
    

def generate_word_cloud(text: str, title: str) -> Any:
    wordcloud = WordCloud(
        scale=3,
        width=500,
        height=330,
        max_words=200,
        colormap='Dark2',
        stopwords=add_multilingual_stopwords(),
        collocations=True,
        contour_color='#5d0f24',
        contour_width=3,
        background_color="rgba(255, 255, 255, 0)").generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    st.pyplot()


def display_statistics(stats):
    total_messages_col, total_members_col, total_media_col = st.columns(3)
    total_messages_col.metric("Total Messages", stats.get('total_messages'))
    total_members_col.metric("Total Members", stats.get('total_members'))
    total_media_col.metric("Total Media", stats.get('media_message'))

    st.text("")

    link_shared_col, deleted_messages_col, your_deleted_messages_col = st.columns(3)
    link_shared_col.metric("Link Shared", int(stats.get('link_shared')))
    deleted_messages_col.metric("Total Deleted Messages", stats.get('total_deleted_messages'))
    # your_deleted_messages_col.metric("Your Deleted Messages", stats.get('your_deleted_message'))

    st.text("")



def chart_display(data_frame):
    st.markdown("----")
    st.header("Over Time Analysis")
    st.warning("Warning: This visualization is optimized for Xiaomi device data and may not work as expected for other datasets.")
    st.write(time_series_plot(data_frame))

    st.markdown("----")
    st.header("Most Active Member")
    st.pyplot(most_active_member(data_frame))

    st.markdown("----")
    st.header("Most Active Day of the Week")
    st.pyplot(most_active_day(data_frame))

    st.markdown("----")
    st.header("Most Words Per Message")
    st.pyplot(max_words_used(data_frame))

    st.markdown("----")
    st.header("Most links Shared")
    st.pyplot(who_shared_links(data_frame))

    st.markdown("----")
    st.header("Most Active Day")
    st.pyplot(most_suitable_day(data_frame))

    st.markdown("----")
    st.header("Most Active Hour")
    st.pyplot(most_suitable_hour(data_frame))
    
    st.markdown("----")
    st.header("Emoji Usage")
    pie_display = pie_display_emojis(data_frame)
    st.plotly_chart(pie_display)
    

def file_process(data, config):
    source_config = WhatsAppConfig(**config['whatsapp'])
    whatsapp = WhatsAppProcess(source_config)
    message = whatsapp.apply_regex(data)
  
    raw_df = process_data(message)
    data_frame = whatsapp.get_dataframe(raw_df)
    stats = statistics(raw_df, data_frame)
    
    st.markdown("----")
    display_statistics(stats)
    
    cloud_df = whatsapp.cloud_data(raw_df)
    
    st.header("Sentment Analysis")
    col4, col5, col6 = st.columns(3)
    

    sentiment_counts = sentiment_analysis_ind(cloud_df) 
    col4.metric("Positive Messages", sentiment_counts['pos_count'])
    col5.metric("Negative Messages", sentiment_counts['neg_count'])
    col6.metric("Neutral Messages", sentiment_counts['neu_count'])
            
    
    st.header("Individual Stats")
    sorted_authors = sorted_authors_df(cloud_df)
    select_author = []
    select_author.append(st.selectbox('', sorted_authors))
    dummy_df = cloud_df[cloud_df['name'] == select_author[0]]
    text = " ".join(review for review in dummy_df.message)
    
    col1, col2, col3 = st.columns(3)
    
    # First row metrics
    col1.metric("Total Messages", dummy_df[dummy_df['name'] == select_author[0]].shape[0])
    col2.metric("Emoji Count", sum(data_frame[data_frame.name.str.contains(select_author[0][-5:])].emojis.str.len()))
    col3.metric("Link Shared", int(data_frame[data_frame.name == select_author[0]].urlcount.sum()))
    

    col7, col8 = st.columns(2)
    col7.metric("Total Words", int(data_frame[data_frame.name == select_author[0]].word_count.sum()))
    user_df = data_frame[data_frame.name.str.contains(select_author[0][-5:])]
    average = round(npsum(user_df.word_count)/user_df.shape[0], 2)
    col8.metric("Average words/Message", average)
    
   
    

    if len(text) != 0:
        generate_word_cloud(
            text, "")
    else:
        generate_word_cloud(
            "NOWORD", "")

    st.markdown("----")
    st.header("Chat Wordcloud")
    text = " ".join(review for review in cloud_df.message)
    generate_word_cloud(
        text, "")
    whatsapp.day_analysis(data_frame)
    chart_display(data_frame)

    st.markdown("----")
    st.header("Media Contributor ")
    st.pyplot(top_media_contributor(raw_df))

    st.markdown("----")
    st.header("Sentiment Analysis")
    st.pyplot(sentiment_analysis(cloud_df))

    st.header("Experimental Features")
    st.warning("Warning: These features are not fully implemented and may cause errors or break the app. Use at your own risk.")
    
    st.markdown("----")
    
    with st.expander("User Activity Over Time"):
        st.pyplot(user_activity_over_time(data_frame))
        
    with st.expander("Test Your Sentiment"):
        vectorizer = pickle.load(open("processor/model/vectorizer1.pickle", "rb"))
        model = pickle.load(open("processor/model/model_1.pickle", "rb"))

        user_input = st.text_input("Enter your sentence here:")

        if user_input:
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)

            if prediction == 0:
                st.write("The sentiment of the text is Negative.")
            elif prediction == 1:
                st.write("The sentiment of the text is Neutral.")
            else:
                st.write("The sentiment of the text is Positive.")

    with st.expander("Search Word"):
        search_terms = st.text_input("Enter your word here:")
        
        if search_terms:
            if " " in search_terms.strip():
                st.error("Please enter only a single word as the search term.")
            else:
                st.pyplot(word_frequency_count(data_frame,[search_terms]))
    
   
def main():
    config = 'configs/app_configuration.yml'
    config = yaml.safe_load(open(config))
    c1, c2 = st.columns([3, 1])
    uploaded_file = c1.file_uploader(
        "Choose a TXT file only",
        type=['txt'],
        accept_multiple_files=False)
    if uploaded_file is not None:
        data = uploaded_file.getvalue().decode("utf-8")
        file_process(data, config)
    st.header("Whatsgraph💬 V4.0 log")
    st.text('''
    version 4 of Whatsgraph contains graphing functions and sentiment analysis.
    In version 5 we would like to do toxicity analysis and improve on the graphing 
    capabilities. 
    
    Additionally, we plan to add more machine learning
    features to enhance data analysis. 
    
    We want to optimize support for other devices other than Xiaomi,
    so please provide us with feedback on any problems you face while 
    using Whatsgraph on different devices.
    ''')
    st.text(''' 
    Major Project
    built by
    Saksham Gorey - 19124043
    Shreesh Bhardwaj - 19124051
    
    ''')
if __name__ == "__main__":
    main()
