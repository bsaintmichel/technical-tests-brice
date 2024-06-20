import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from default_checker.model import load_model, load_preproc, preprocess_for_evaluation
from pydantic import BaseModel
from numpy import int, float, nan

app = FastAPI(debug=True)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class Data(BaseModel):
    uuid : list[str]
    account_amount_added_12_24m : list[int]
    account_days_in_dc_12_24m : list[float]
    account_days_in_rem_12_24m : list[float]
    account_days_in_term_12_24m : list[float]
    account_incoming_debt_vs_paid_0_24m : list[float]
    account_status : list[float]
    account_worst_status_0_3m : list[float]
    account_worst_status_12_24m : list[float]
    account_worst_status_3_6m : list[float]
    account_worst_status_6_12m : list[float]
    age : list[int]
    avg_payment_span_0_12m : list[float]
    avg_payment_span_0_3m : list[float]
    merchant_category : list[str]
    merchant_group : list[str]
    has_paid : list[bool]
    max_paid_inv_0_12m : list[float]
    max_paid_inv_0_24m : list[float]
    num_active_div_by_paid_inv_0_12m : list[float]
    num_active_inv : list[int]
    num_arch_dc_0_12m : list[int]
    num_arch_dc_12_24m : list[int]
    num_arch_ok_0_12m : list[int]
    num_arch_ok_12_24m : list[int]
    num_arch_rem_0_12m : list[int]
    num_arch_written_off_0_12m : list[float]
    num_arch_written_off_12_24m : list[float]
    num_unpaid_bills : list[int]
    status_last_archived_0_24m : list[int]
    status_2nd_last_archived_0_24m : list[int]
    status_3rd_last_archived_0_24m : list[int]
    status_max_archived_0_6_months : list[int]
    status_max_archived_0_12_months : list[int]
    status_max_archived_0_24_months : list[int]
    recovery_debt : list[int]
    sum_capital_paid_account_0_12m : list[int]
    sum_capital_paid_account_12_24m : list[int]
    sum_paid_inv_0_12m : list[int]
    time_hours : list[float]
    worst_status_active_inv : list[float]

app.state.model = load_model()
app.state.preproc = load_preproc()

@app.post("/predict")
async def predict(df_in: Data):
    '''As you probably see, the data will not be validated at all ...'''

    data = pd.read_json(df_in.json())
    data = data.replace(-99999, nan)
    data_pp, uuids = preprocess_for_evaluation(data, app.state.preproc)
    probas = app.state.model.predict_proba(data_pp)

    return {'uuid': list(uuids),
            'pp': list(probas[:,0])}

@app.get("/")
def root():
    return {'greeting':'Hello'}
