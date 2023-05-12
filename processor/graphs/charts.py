from typing import Any
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import matplotlib.pyplot as plt
from typing import List


def pie_display_emojis(data_frame: pd.DataFrame):
    total_emojis_list = list(set([a for b in data_frame.emojis for a in b]))
    total_emojis_list = (a for b in data_frame.emojis for a in b)
    emoji_dict = dict(Counter(total_emojis_list))
    emoji_dict = sorted(
        emoji_dict.items(), key=lambda x: x[1], reverse=True)

    emoji_df = pd.DataFrame(emoji_dict, columns=['emojis', 'count'])
    fig = px.pie(emoji_df, values='count', names='emojis')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def time_series_plot(data_frame: pd.DataFrame):
    z_value = data_frame['date'].value_counts()
    z_dict = z_value.to_dict()  
    data_frame['msg_count'] = data_frame['date'].map(z_dict)
    fig = px.line(x=data_frame['date'], y=data_frame['msg_count'])
    fig.update_layout(
        xaxis_title='Time Stamp',
        yaxis_title='Number of Messages')
    fig.update_xaxes(nticks=60)
    fig.update_traces(line_color='#25D366')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_data(data_string):
    fig, ax_value = plt.subplots(facecolor='none')
    bars = ax_value.bar(
        x=np.arange(data_string.get('x_value')),
        height=data_string.get('y_value'),
        tick_label=data_string.get('tick_label'),
        color="#25D366"
    )
    ax_value.set_xticklabels(data_string.get('tick_label'), fontfamily='monospace')
    ax_value.set_yticklabels(ax_value.get_yticks(), fontfamily='monospace')
    ax_value.set_xlabel(
        data_string.get('x_label'), labelpad=15, color='#333333', fontfamily='monospace')
    ax_value.set_ylabel(
        data_string.get('y_label'), labelpad=15, color='#333333', fontfamily='monospace')
    ax_value.set_title(
        data_string.get('title'), pad=15, color='#333333', fontfamily='monospace')
    ax_value.spines['top'].set_visible(False)
    ax_value.spines['right'].set_visible(False)
    ax_value.spines['left'].set_visible(False)
    ax_value.spines['bottom'].set_color('#25D366')
    ax_value.tick_params(bottom=False, left=False)
    ax_value.tick_params(axis='x', labelrotation=90)
    ax_value.set_axisbelow(True)
    ax_value.yaxis.grid(True, color='#EEEEEE')
    ax_value.xaxis.grid(False)

    return fig


def max_words_used(data_frame: pd.DataFrame):
    max_words = data_frame[['name', 'word_count']].groupby('name').sum()
    m_w = max_words.sort_values('word_count', ascending=False).head(10)
    return plot_data({
            'x_value': m_w.size,
            'y_value': m_w.word_count,
            'tick_label': m_w.index,
            'x_label': 'Name of Member',
            'y_label': 'Number of Words in Chat',
        })



def most_active_member(data_frame: pd.DataFrame):
    mostly_active = data_frame['name'].value_counts()
    m_a = mostly_active.head(10)
    return plot_data({
            'x_value': m_a.size,
            'y_value': m_a,
            'tick_label': m_a.index,
            'x_label': 'Name of Member',
            'y_label': 'Number of Messages',
        })


def most_active_day(data_frame: pd.DataFrame):
    active_day = data_frame['day'].value_counts()
    a_d = active_day.head(10)
    return plot_data({
            'x_value': a_d.size,
            'y_value': a_d,
            'tick_label': a_d.index,
            'x_label': 'Name of Member',
            'y_label': 'Number of Messages',
           
        })


def top_media_contributor(data_frame: pd.DataFrame):
    max_media = data_frame[['name', 'media']].groupby('name').sum()
    m_m = max_media.sort_values(
        'media', ascending=False).head(10)
    return plot_data({
            'x_value': m_m.size,
            'y_value': m_m.media,
            'tick_label': m_m.index,
            'x_label': 'Name of Member',
            'y_label': 'Number of Media',
        })


def who_shared_links(data_frame: pd.DataFrame):
    # Member who has shared max numbers of link in Group
    max_words = data_frame[['name', 'urlcount']].groupby('name').sum()
    m_w = max_words.sort_values('urlcount', ascending=False).head(10)
    return plot_data({
            'x_value': m_w.size,
            'y_value': m_w.urlcount,
            'tick_label': m_w.index,
            'x_label': 'Name of Group Member',
            'y_label': 'Number of Links Shared',
        })


def time_when_group_active(data_frame: pd.DataFrame):
    active_time = data_frame.datetime.dt.time.value_counts().head(10)
    return plot_data({
            'x_value': active_time.size,
            'y_value': active_time.values,
            'tick_label': active_time.index,
            'x_label': 'Time',
            'y_label': 'Number of Messages',
        })


def most_suitable_hour(data_frame: pd.DataFrame):
    active_hour = data_frame.datetime.dt.hour.value_counts().head(20)
    tick_labels = [pd.to_datetime(str(hour), format='%H').strftime('%I %p') for hour in active_hour.index]
    return plot_data({
            'x_value': active_hour.size,
            'y_value': active_hour.values,
            'tick_label': tick_labels,
            'x_label': 'Hour',
            'y_label': 'Number of Messages',
        })

def word_frequency_count(data_frame: pd.DataFrame, search_terms: List[str]):
    # Combine all messages into a single string
    messages = data_frame['message'].str.cat(sep=' ')
    
    term_counts = {}
    for term in search_terms:
        count = 0
        for index, row in data_frame.iterrows():
            if term in row['message']:
                name = row['name']
                count += row['message'].count(term)
                if name in term_counts:
                    term_counts[name] += row['message'].count(term)
                else:
                    term_counts[name] = row['message'].count(term)
        if count > 0:  # Only add term if it is found at least once
            term_counts[term] = count
    
    # Get unique member names from 'name' column
    member_names = list(set(data_frame['name']))
    
    # Create bar chart of results
    fig, ax = plt.subplots()
    ax.bar(member_names, [term_counts.get(name, 0) for name in member_names], color='#25D366')  # Use dictionary.get method to return 0 for missing keys
    ax.set_xlabel('Member Name')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(member_names, rotation=45)  # Set x-tick labels to member names
    
    return fig



def most_suitable_day(data_frame: pd.DataFrame):
    active_day = data_frame.datetime.dt.day.value_counts().head(20)
    x_labels = [f"{day}{' th' if day in [11, 12, 13] else {1:'st', 2:'nd', 3:'rd'}.get(day % 10, 'th')}" for day in active_day.index]
    return plot_data({
            'x_value': active_day.size,
            'y_value':  active_day.values,
            'tick_label': x_labels,
            'x_label': 'Day',
            'y_label': 'Number of Messages',
        })


def sentiment_analysis(cloud_df: pd.DataFrame):
    cloud_df['sentiment'] = cloud_df.message.apply(
        lambda text: TextBlob(text).sentiment.polarity)
    sentiment = cloud_df[['name', 'sentiment']].groupby('name').mean()
    s_a = sentiment.sort_values('sentiment', ascending=False).head(10)
    return plot_data({
            'x_value': s_a.size,
            'y_value': s_a.sentiment,
            'tick_label': s_a.index,
            'x_label': 'Name of Member',
            'y_label': 'Positive Sentiment',
        })


def user_activity_over_time(data_frame: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12,6))
    
    for name in data_frame['name'].unique():
        member_data = data_frame[data_frame['name'] == name]
        activity_counts = member_data['datetime'].dt.floor('H').value_counts()
        activity_counts.sort_index(inplace=True)
        
        ax.plot(activity_counts.index, activity_counts.values, label=name)

    ax.set_xlabel('Time (hourly)')
    ax.set_ylabel('Number of Messages')
    ax.set_title('User Activity Over Time')
    ax.legend()
    
    return fig



