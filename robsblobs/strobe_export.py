import numpy as np
import pandas as pd
import glob
import scipy.io as sio

from .monitor import Monitor
from .cie_monitor_helpers import rgb2xyz

flickerTypes = dict([
    (0, "RedGreen"),
    (1, "RedPurple"),
    (2, "GreenPurple"),
    (3, "GrayGreen"),
    (4, "GrayPurple"),
    (5, "GrayYellow"),
    (6, "GreenYellow"),
    (7, "GrayRed"),
    (8, "GrayBlue"),
    (9, "GrayCyan")
])

motionTypes = dict([
    (0, "RedGreen"),
    (1, "GreenYellow"),
    (2, "PurpleGreen"),
    (3, "GreenCyan")
])

def convert_color_weight_to_lum(mon, df):
    w = df['Weight']
    c = df['Color'][0]

    rgb_gc = np.zeros(3)
    rgb = np.zeros(3)
    for rc in range(3):
        # because this is what we sent to the monitor
        rgb_gc[rc] = np.power(w * c[rc], 1.0/2.2)
        rgb[rc] = np.power(rgb_gc[rc], mon.monGamma[rc])
            
    xyz = rgb2xyz(mon, rgb)
    return xyz[1]


def parse_flicker_data():
    pass
    #         for c in range(2):
    #             t = pd.DataFrame({
    #                 "StimType": "Flicker",
    #                 "SubjectName": str(db.iloc[tc]["SubjectName"]),
    #                 "BlockNumber": db.iloc[tc]["BlockNumber"],
    #                 "TrialInBlock": db.iloc[tc]["TrialInBlock"],
    #                 "TotalTrials": db.iloc[tc]["TotalTrials"],
    #                 "ReactionTime": float(db.iloc[tc]["ReactionTime"]),
    #                 "Flick": "First" if c == 0 else "Second",
    #                 "Weight": float(resp['weight_first']) if c == 0 else float(resp['weight_second']),
    #                 "Color": [resp['first_col'] if c == 0 else resp['second_col']],
    #                 "FlickerType": flickerTypes[resp['flicker_type']]
    #             })
    #             df = pd.concat([df, t])


def parse_minmotion_data(mon: Monitor, data, resp, tc):
    t = []
    for c in range(2):
        df = pd.DataFrame({
            "StimType": "MinimalMotion",
            "SubjectName": data["SubjectName"][tc],
            "BlockNumber": data["BlockNumber"][tc],
            "TrialInBlock": data["TrialInBlock"][tc],
            "TotalTrials": data["TotalTrials"][tc],
            "ReactionTime": data["ReactionTime"][tc],
            "ColorComponent": "First" if c == 0 else "Second",
            "Weight": float(resp['weight_first'] if c == 0 else resp['weight_second']),
            "Color": [resp['first_col'] if c == 0 else resp['second_col']],
            "MotionType": motionTypes[resp['motion_type']]
        })

        df["Luminances"] = convert_color_weight_to_lum(mon, df)

        t.append(df)

    return t
    

def parse_unique_yellow_data():
    pass
    # t = pd.DataFrame({
    #         "StimType": "UniqueYellow",
    #         "SubjectName": str(db.iloc[tc]["SubjectName"]),
    #         "BlockNumber": db.iloc[tc]["BlockNumber"],
    #         "TrialInBlock": db.iloc[tc]["TrialInBlock"],
    #         "TotalTrials": db.iloc[tc]["TotalTrials"],
    #         "ReactionTime": float(db.iloc[tc]["ReactionTime"]),
    #         "WeightR": [float(resp["weight_R"])],
    #         "WeightG": [float(resp["weight_G"])]
    #     })
    #     df = pd.concat([df, t])


def parse_brightness_sorting_data(mon: Monitor, data, resp, tc):
    indexes = []
    rs = []
    gs = []
    bs = []
    for rc in range(len(resp)):
        indexes.append(resp[rc]["rect_index"])
        rs.append(float(resp[rc]["color"][0]))
        gs.append(float(resp[rc]["color"][1]))
        bs.append(float(resp[rc]["color"][2]))

    t = pd.DataFrame({
        "StimType": "BrightnessSorting",
        "SubjectName": str(data.iloc[tc]["SubjectName"]),
        "BlockNumber": data.iloc[tc]["BlockNumber"],
        "TrialInBlock": data.iloc[tc]["TrialInBlock"],
        "TotalTrials": data.iloc[tc]["TotalTrials"],
        "ReactionTime": float(data.iloc[tc]["ReactionTime"]),
        "RectangleIndex": [indexes],
        "R": [rs],
        "G": [gs],
        "B": [bs],
    })
    
    return t


def parse_strobe_data(mon: Monitor, fn):
    d = pd.read_json(fn)

    d = d.rename({
        0: "SubjectName",
        1: "BlockNumber",
        2: "TrialInBlock",
        3: "TotalTrials",
        4: "ObserverResponse",
        5: "ReactionTime",
        6: "DontKnow",
    }, axis=1)
    d = d.drop(columns=["DontKnow"])

    sname = str(d["SubjectName"][0])

    df = pd.DataFrame()
    for tc in d["TotalTrials"]:
        resp = d.iloc[tc]["ObserverResponse"]
        if len(resp) == 0:
            continue

        ts = []
        if 'motion_type' in resp:
            ts = parse_minmotion_data(mon, d, resp, tc)
        elif 'weight_R' in resp:
            continue
            ts = parse_unique_yellow_data(mon, d, resp, tc)
        elif 'rect_index' in resp[0]:
            ts = parse_brightness_sorting_data(mon, d, resp, tc)
        elif 'flicker_type' in resp:
            continue
            ts = parse_flicker_data(mon, d, resp, tc)

        if isinstance(ts, list) and len(ts) > 0:
            for t in ts:
                df = pd.concat([df, t])
        else:
            df = pd.concat([df, ts])
        
    df = df.set_axis([str(k) for k in range(len(df))], axis='index')

    return df, sname


def export_strobe_data_to_json(fn, df):
    out = df.to_json(orient='records')
    with open(fn, "w") as f:
        f.write(out)


def export_strobe_data_to_matlab(fn, df):
    sio.savemat(fn, {'data': df.to_dict("list")})