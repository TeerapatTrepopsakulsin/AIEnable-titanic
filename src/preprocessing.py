import pandas as pd


# Load data
df = pd.read_csv('../data/train.csv')

# Drop
df.drop(['Name', 'Cabin'], axis=1, inplace=True)

# Encoding
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Feature Engineering
split_ticket_val = df["Ticket"].str.split(" ")
df["TicketNumber"] = split_ticket_val.str[-1]
df["TicketPrefix"] = split_ticket_val.str[:-1].apply(lambda x: "".join(x) if x else "")

df["HasPrefix"] = df["TicketPrefix"].apply(lambda x: 1 if x else 0)
df["TicketLength"] = df["TicketNumber"].astype(str).apply(len)
df["TicketIsLine"] = df["TicketNumber"].apply(lambda x: 1 if str(x).upper() == "LINE" else 0)
df["TicketNumber"] = pd.to_numeric(df["TicketNumber"], errors="coerce").fillna(0).astype(int)

# Drop
df.drop(['Ticket', 'TicketPrefix'], axis=1, inplace=True)

df.to_csv('../data/train_preprocessed.csv', index=False)
print(df.head())
