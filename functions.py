import json

def conll_df_to_conll_str(conll_df):

    curr_conll_doc_key = ""
    conll_str = ""

    for i in range(conll_df.shape[0]):
        row = conll_df.iloc[i,:]
        conll_str += "\n"+ row["conll_doc_key"] + "\t" + str(row["sent_id"]) + \
                           "\t" + str(row["token_id"]) + "\t" + row["token"] + "\t" + row["reference"]

    return conll_str



def mentions_df_to_json_list(mentions_df):

    mentions_json = []    
    #for mention in mentions_df
    for index,row in mentions_df.iterrows():
        mention_json = json.loads(row.to_json())
        mentions_json.append(mention_json)
        


    return mentions_json
