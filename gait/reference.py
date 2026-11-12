"""画面に出す「目安」と出典。出典のない数値はここに置かない。"""

REFERENCES = {
    "stride_time_cv": {
        "note": "健康な方の定常歩行では、ストライド時間のばらつき(CV)はおおむね数%以内と報告されています"
                "（メタ分析で上限の目安 2.6%）。動画からの計測は誤差を含むため、数値そのものより"
                "「同じ条件で撮った前回との違い」を見るのがおすすめです。",
        "source": "König et al. 2016 / Kroneberg et al. 2019, Front Aging Neurosci. doi:10.3389/fnagi.2018.00435",
    },
    "min_strides": {
        "value": 15,
        "note": "ばらつきの評価には、15ストライド程度以上が目安とされています。",
        "source": "Kroneberg et al. 2019, Front Aging Neurosci. doi:10.3389/fnagi.2018.00435",
    },
    "pose_accuracy": {
        "note": "スマホ動画＋MediaPipeの膝角度は、基準機器と比べて平均で数度の誤差が報告されています。"
                "数度の違いは誤差の範囲と考えてください。",
        "source": "Knee angle with MediaPipe Pose vs Kinovea (MAE 5.88°), IOP Conf. Ser.: Mater. Sci. Eng.",
    },
    "workers_gait": {
        "note": "働く世代を対象に、歩き始めの動画とMediaPipeで転倒リスクとの関連を調べた研究があります"
                "（男性では高い判別精度、女性では十分とは言えない精度）。",
        "source": "Sakane et al. 2025, J Occup Health 67(1):uiae075",
    },
}
