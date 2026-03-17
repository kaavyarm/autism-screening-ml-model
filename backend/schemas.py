from pydantic import BaseModel


class PredictionRequest(BaseModel):
    A1_Score: int
    A2_Score: int
    A3_Score: int
    A4_Score: int
    A5_Score: int
    A6_Score: int
    A7_Score: int
    A8_Score: int
    A9_Score: int
    A10_Score: int

    age: int
    gender: int       # 1 = male, 0 = female
    jundice: int      # 1 = yes, 0 = no
    austim: int       # 1 = yes, 0 = no

    # optional categorical fields
    ethnicity: str = "Other"
    contry_of_res: str = "Unknown"