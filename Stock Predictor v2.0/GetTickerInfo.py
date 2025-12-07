import requests

def get_primary_ns_ticker(query):
    url = "https://query1.finance.yahoo.com/v1/finance/search"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    params = {"q": query, "lang": "en-US", "region": "US", "quotesCount": 20}
    
    r = requests.get(url, headers=headers, params=params, timeout=10)
    try:
        data = r.json()
    except:
        return None
    
    quotes = data.get("quotes", [])
    ns_list = []

    for q in quotes:
        symbol = q.get("symbol", "")
        name = q.get("shortname", "")

        if symbol.endswith(".NS"):
            ns_list.append(symbol)

    if not ns_list:
        return None
    
    primary = [s for s in ns_list if "-" not in s.split(".")[0]]

    if primary:
        return primary[0]
    else:
        return ns_list[0]
    

company = input("Enter company name: ")
ticker = get_primary_ns_ticker(company)

print(ticker)