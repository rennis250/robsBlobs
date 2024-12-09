import numpy as np
import pandas as pd
import glob
import scipy.io as sio

from .monitor import Monitor
from .cie_monitor_helpers import rgb2xyz
from .dkl import dkl2rgb, rgb2dkl
from .infamous_lab import xyz2lab

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
        rgb_gc[rc] = np.power(w.iloc[0] * c[rc], 1.0/2.2)
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
    

def parse_unique_yellow_data(mon: Monitor, data, resp, tc):
    t = pd.DataFrame({
        "StimType": "UniqueYellow",
        "SubjectName": str(data.iloc[tc]["SubjectName"]),
        "BlockNumber": data.iloc[tc]["BlockNumber"],
        "TrialInBlock": data.iloc[tc]["TrialInBlock"],
        "TotalTrials": data.iloc[tc]["TotalTrials"],
        "ReactionTime": float(data.iloc[tc]["ReactionTime"]),
        "WeightR": [float(resp["weight_R"])],
        "WeightG": [float(resp["weight_G"])]
    })

    return t


def parse_uy_elim_data(mon: Monitor, data, resp, tc):
    indexes = []
    left_over = []
    rs = []
    gs = []
    bs = []
    wrs = []
    wgs = []
    for rc in range(len(resp)):
        w = resp[rc]["color"]
        rgb = [w, 1.0 - w, 0.0]
        wr = w
        wg = 1.0 - w

        indexes.append(resp[rc]["rect_index"])
        rs.append(float(rgb[0]))
        gs.append(float(rgb[1]))
        bs.append(float(rgb[2]))
        left_over.append(resp[rc]["left_over"])
        wrs.append(wr)
        wgs.append(wg)
        
    t = pd.DataFrame(
        {
            "StimType": "UniqueYellowEliminator",
            "SubjectName": str(data.iloc[tc]["SubjectName"]),
            "BlockNumber": data.iloc[tc]["BlockNumber"],
            "TrialInBlock": data.iloc[tc]["TrialInBlock"],
            "TotalTrials": data.iloc[tc]["TotalTrials"],
            "ReactionTime": float(data.iloc[tc]["ReactionTime"]),
            "RectangleIndex": [indexes],
            "LeftOver": [left_over],
            "R": [rs],
            "G": [gs],
            "B": [bs],
            "WeightR": [wrs],
            "WeightG": [wgs],
        }
    )

    return t


def parse_ws_elim_data(mon: Monitor, data, resp, tc):
    indexes = []
    left_over = []
    rs = []
    gs = []
    bs = []
    ls = []
    ass = []
    lbs = []
    for rc in range(len(resp)):
        indexes.append(resp[rc]["rect_index"])
        rs.append(float(resp[rc]["color"][0]))
        gs.append(float(resp[rc]["color"][1]))
        bs.append(float(resp[rc]["color"][2]))
        left_over.append(resp[rc]["left_over"])

        r = float(resp[rc]["color"][0])
        g = float(resp[rc]["color"][1])
        b = float(resp[rc]["color"][2])

        r = r / 255 # if r > 1 else r
        g = g / 255 # if g > 1 else g
        b = b / 255 # if b > 1 else b

        ldrgyv = rgb2dkl(mon, np.array([r, g, b]).T)

        ld = ldrgyv[0]
        rg = ldrgyv[1]
        yv = ldrgyv[2]

        rgb = dkl2rgb(mon, np.array([ld, rg, yv]).T)
        xyz = rgb2xyz(mon, rgb)
        lab = xyz2lab(mon, xyz)

        ls.append(lab[0])
        ass.append(lab[1])
        lbs.append(lab[2])

    t = pd.DataFrame(
        {
            "StimType": "WhiteSettingEliminatorDKL_small_extent",
            "SubjectName": str(data.iloc[tc]["SubjectName"]),
            "BlockNumber": data.iloc[tc]["BlockNumber"],
            "TrialInBlock": data.iloc[tc]["TrialInBlock"],
            "TotalTrials": data.iloc[tc]["TotalTrials"],
            "ReactionTime": float(data.iloc[tc]["ReactionTime"]),
            "RectangleIndex": [indexes],
            "LeftOver": [left_over],
            "R": [rs],
            "G": [gs],
            "B": [bs],
            "L": [ls],
            "a": [ass],
            "b": [lbs]
        }
    )

    return t


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


def parse_white_setting_data(mon: Monitor, data, resp, tc):
    t = pd.DataFrame()

    ld = resp["ld"]
    rg = resp["rg"]
    yv = resp["yv"]

    rgb = dkl2rgb(mon, [ld, rg, yv])
    xyz = rgb2xyz(mon, rgb)
    lab = xyz2lab(mon, xyz)

    t = pd.DataFrame({
        "StimType": "WhiteSetting",
        "SubjectName": str(data.iloc[tc]["SubjectName"]),
        "BlockNumber": data.iloc[tc]["BlockNumber"],
        "TrialInBlock": data.iloc[tc]["TrialInBlock"],
        "TotalTrials": data.iloc[tc]["TotalTrials"],
        "ReactionTime": float(data.iloc[tc]["ReactionTime"]),
        "LD": ld,
        "RG": rg,
        "YV": yv,
        "R": rgb[0],
        "G": rgb[1],
        "B": rgb[2],
        "L": lab[0],
        "a": lab[1],
        "b": lab[2]
    }, index = [1])

    return t


def parse_mult_choice_data(mon: Monitor, data, resp, tc):
    t = pd.DataFrame({
        "StimType": "MultipleChoice",
        "SubjectName": str(data.iloc[tc]["SubjectName"]),
        "BlockNumber": data.iloc[tc]["BlockNumber"],
        "TrialInBlock": data.iloc[tc]["TrialInBlock"],
        "TotalTrials": data.iloc[tc]["TotalTrials"],
        "ReactionTime": float(data.iloc[tc]["ReactionTime"]),
        "CorrResp": resp['corr_resp'],
        "ObsResp": resp['obs_resp'] - 48
    }, index = [1])

    return t


def parse_strobe_data(mon: Monitor, fn, exp_json):
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
    # for tc in d["TotalTrials"]:
    for tc in range(len(d)):
        # bc = d.iloc[tc]["BlockNumber"]
        # tib = d.iloc[tc]["TrialInBlock"]
        resp = d.iloc[tc]["ObserverResponse"]
        if len(resp) == 0:
            continue

        # stim_type = exp_json["blocks"][bc]["trials"][tib]["stages"][0]["type"]

        ts = []
        # if stim_type == "WhiteSettingEliminatorDKL_small_extent":
            # ts = parse_ws_elim_data(mon, d, resp, tc)
        # elif stim_type == "UniqueYellowEliminator":
            # ts = parse_uy_elim_data(mon, d, resp, tc)
        # elif stim_type == "MinimalMotionShader" or 'motion_type' in resp:
        if 'motion_type' in resp:
            ts = parse_minmotion_data(mon, d, resp, tc)
        elif 'weight_R' in resp:
            ts = parse_unique_yellow_data(mon, d, resp, tc)
        elif 'flicker_type' in resp:
            continue
            ts = parse_flicker_data(mon, d, resp, tc)
        elif 'ld' in resp:
            ts = parse_white_setting_data(mon, d, resp, tc)
        elif 'corr_resp' in resp and 'obs_resp' in resp:
            ts = parse_mult_choice_data(mon, d, resp, tc)
        elif len(resp) > 1:
            if len(resp) > 1:
                # print(resp)
                if isinstance(resp[0]["color"], float):
                    ts = parse_uy_elim_data(mon, d, resp, tc)
                else:
                    ts = parse_ws_elim_data(mon, d, resp, tc)
            # if 'rect_index' in resp[0]:
                # ts = parse_brightness_sorting_data(mon, d, resp, tc)
        
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