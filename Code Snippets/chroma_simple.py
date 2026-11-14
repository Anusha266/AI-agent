import chromadb

chroma=chromadb.Client()

zorvex_collection=chroma.create_collection(
    name="zorvex_technology"
)

zorvex_collection.add(
    ids=["c0","c1","c2","c3"],
    documents=[
        "Leave Policy - Maternity Leave\nMaternity leave is 26 weeks of paid leave for the first two children.",
        "Medical Policy - Maternity Cover\nHospitalization for childbirth is covered up to 75,000 rupees.",
        "Travel Expense - Reimbursement\nSubmit expense claims within 30 days of the trip. Approved claims are reimbursed within 7 working days.",
        "Tech Docs - Deployment Windows\nProduction deployments are allowed on Tuesdays and Thursdays between 11 AM and 4 PM.",
    ]
)

result = zorvex_collection.query(
    query_texts=["when is the production deployment allowed"],
    n_results=1
)

print(result)