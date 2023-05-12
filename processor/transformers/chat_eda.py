import re
from typing import List, Dict, Any, NamedTuple
import pandas as pd
import emoji
from pandas.errors import EmptyDataError


def get_members(data_frame: pd.DataFrame) -> pd.DataFrame:
    return data_frame["name"].unique()


def sorted_authors_df(data_frame: pd.DataFrame) -> List:
    sorted_authors = data_frame.groupby('name')['message'].count()\
        .sort_values(ascending=False).index
    return sorted_authors


def statistics(raw_df: pd.DataFrame, data_frame: pd.DataFrame) -> Dict:
    author_list = get_members(data_frame)
    return {
        "media_message": len(
            raw_df[raw_df.message.str.contains("omitted")]),
        "total_deleted_messages": len(
            raw_df[raw_df['message'] == "This message was deleted"]),
        # "your_deleted_message": len(
        #     raw_df[raw_df['message'] == "You deleted this message"]),
        "group_name": raw_df.iloc[0:1]['name'][0],
        "total_messages": data_frame.shape[0],
        'total_members': len(author_list),
        'link_shared': data_frame['urlcount'].sum()}


def extract_emojis(string: str) -> str:
    return ''.join(c for c in string if c in emoji.UNICODE_EMOJI['en'])


def give_emoji_free_text(text: str) -> str:
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text


def process_data(messages: str) -> pd.DataFrame:
    
    raw_df = pd.DataFrame(
        messages, columns=['datetime', 'name', 'message'])
    
    # SAMSUNG Export time format
    try:
        raw_df['datetime'] = raw_df['datetime'].str.replace(
            r'[p].[m].', 'PM', regex=True)
        raw_df['datetime'] = raw_df['datetime'].str.replace(
            r'[a].[m].', 'AM', regex=True)
        raw_df['datetime'] = pd.to_datetime(
            raw_df['datetime'], format="%Y-%m-%d, %I:%M %p")
        
    except ValueError:
        pass
    
    # IOS Export time format
    try:
        # Drop date enclosures from date column
        raw_df['datetime'] = raw_df['datetime'].map(
            lambda x: x.lstrip('[').rstrip(']'))
        raw_df['datetime'] = pd.to_datetime(
            raw_df['datetime'], format="%m/%d/%y, %I:%M %p")
        
    except ValueError:
        pass

    # OppO Export time format
    try:
        raw_df['datetime'] = pd.to_datetime(
            raw_df['datetime'], format="%d/%m/%Y, %I:%M %p")
        
    except ValueError:
        pass

    # Android Export time format
    try:
        raw_df['datetime'] = pd.to_datetime(
            raw_df['datetime'], format="%d/%m/%y, %I:%M %p")
        
    except ValueError:
        pass

    raw_df['date'] = pd.to_datetime(raw_df['datetime']).dt.date
    raw_df['time'] = pd.to_datetime(raw_df['datetime']).dt.time
    
    return raw_df



class WhatsAppConfig(NamedTuple):
    url_pattern: str
    weeks: dict
    regex_list: list
    ignore: list 


class WhatsAppProcess():
    def __init__(self, app_config):
        self.app_config = app_config
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)

    def apply_regex(self, data: str) -> List:    
        matches = []
        for reg in self.app_config.regex_list:
            matches += re.findall(reg, data)
        return matches

    def get_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        raw_df['media'] = raw_df['message'].apply(
            lambda x: re.findall("omitted", x)).str.len()
        data_frame = raw_df.assign(
            emojis=raw_df["message"].apply(extract_emojis))
        data_frame['urlcount'] = data_frame.message.apply(
            lambda x: re.findall(self.app_config.url_pattern, x)).str.len()
        data_frame['urlcount'].groupby(by=data_frame['name']).sum()
        media_messages_df = data_frame[
            data_frame['message'].str.contains("omitted")]
        messages_df = data_frame.drop(media_messages_df.index)
        messages_df['letter_count'] = messages_df['message'].apply(
            lambda s: len(s))
        messages_df['word_count'] = messages_df['message'].str.len()
        messages_df["message_count"] = 1

        return messages_df

    def day_analysis(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        data_frame.groupby('name')['message'].count()\
            .sort_values(ascending=False).index
        # Make sure datetime column is of dtype datetime64[ns]
        data_frame['datetime'] = pd.to_datetime(data_frame['datetime'])
        data_frame['day'] = data_frame.datetime.dt.weekday.map(self.app_config.weeks)
        # Rearranging the columns for better understanding
        data_frame = data_frame[[
            'datetime', 'day', 'name', 'message',
            'date', 'time', 'emojis', 'urlcount']]
        data_frame.loc[:, 'day'] = data_frame['day'].astype('category')
        return data_frame

    
     

    def cloud_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        sep = '|'
        cloud_df = raw_df[(raw_df["message"].str.contains(
            sep.join(self.app_config.ignore)) == False)]
        modified_df = cloud_df.copy()
        modified_df.message = cloud_df.loc[:, 'message'].apply(
            lambda s: s.lower())\
            .apply(lambda s: self.emoji_pattern.sub(r'', s))\
            .str.replace('\n|\t', '', regex=True)\
            .str.replace(' {2,}', ' ', regex=True)\
            .str.strip().replace(r'http\S+', '', regex=True)\
            .replace(r'www\S+', '', regex=True)
        return modified_df

