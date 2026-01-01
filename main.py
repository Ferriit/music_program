import numpy as np
import sounddevice as sd
import math
import re
import numpy.typing as npt
import sys


wave_points = []

sample_rate = 44100

def fuzz(n):
    return math.copysign(1.0, n) if n != 0 else 0.0

def distortion(n, gain=20.0, threshold=0.7):
    x = n * gain
    return max(min(x, threshold), -threshold) / threshold

def overdrive(n):
    return n / (1 + abs(n))

def clean(n):
    return n

def convert_tone(semitone):
    """Converts a measure of semitones to hz. A0 = 0, A1 = 12, A2 = 24, A3 = 36, A4 = 48"""

    return 55 * math.pow(2, semitone / 12)


def all_in_one(n, dist_type):
    match dist_type:
        case "d":
            return distortion(n)
        case "o":
            return overdrive(n)
        case "f":
            return fuzz(n)

    return clean(n)


def tone_sine(tone, length, volume, samplerate, bpm, effects=[]):
    distortion_mode = "c"
    decay_enabled = False
    ring_enabled = False

    mapping_distortion = {"dist": "d", "over": "o", "fuzz": "f", "clean": "c"}
    for eff in effects:
        if eff in mapping_distortion:
            distortion_mode = mapping_distortion[eff]
        elif eff == "decay":
            decay_enabled = True
        elif eff == "ring":
            ring_enabled = True

    seconds_per_beat = 60 / bpm
    duration = length * seconds_per_beat
    sample_length = int(samplerate * duration)

    wave_list = []
    freq = convert_tone(tone)
    for i in range(sample_length):
        t = i / samplerate
        sample = math.sin(2 * math.pi * freq * t)
        sample = all_in_one(sample, distortion_mode)
        sample *= volume / 100
        wave_list.append(sample)

    if decay_enabled:
        n = len(wave_list)
        for i in range(n):
            wave_list[i] *= math.exp(-i / n)

    return np.array(wave_list, dtype=np.float32)


def tone_saw(tone, length, volume, samplerate, bpm, effects=[]):
    distortion_mode = "c"
    decay_enabled = False
    ring_enabled = False

    mapping_distortion = {"dist": "d", "over": "o", "fuzz": "f", "clean": "c"}
    for eff in effects:
        if eff in mapping_distortion:
            distortion_mode = mapping_distortion[eff]
        elif eff == "decay":
            decay_enabled = True
        elif eff == "ring":
            ring_enabled = True

    seconds_per_beat = 60 / bpm
    duration = length * seconds_per_beat
    sample_length = int(samplerate * duration)

    wave_list = []
    freq = convert_tone(tone)
    for i in range(sample_length):
        t = i / samplerate
        sample = 2 * ((t * freq) - math.floor(0.5 + t * freq))
        sample = all_in_one(sample, distortion_mode)
        sample *= volume / 100
        wave_list.append(sample)

    if decay_enabled:
        n = len(wave_list)
        for i in range(n):
            wave_list[i] *= math.exp(-i / n)

    return np.array(wave_list, dtype=np.float32)


def tone_tri(tone, length, volume, samplerate, bpm, effects=[]):
    distortion_mode = "c"
    decay_enabled = False
    ring_enabled = False

    mapping_distortion = {"dist": "d", "over": "o", "fuzz": "f", "clean": "c"}
    for eff in effects:
        if eff in mapping_distortion:
            distortion_mode = mapping_distortion[eff]
        elif eff == "decay":
            decay_enabled = True
        elif eff == "ring":
            ring_enabled = True

    seconds_per_beat = 60 / bpm
    duration = length * seconds_per_beat
    sample_length = int(samplerate * duration)

    wave_list = []
    freq = convert_tone(tone)
    for i in range(sample_length):
        t = i / samplerate
        frac = (t * freq) % 1
        sample = 4 * abs(frac - 0.5) - 1
        sample = all_in_one(sample, distortion_mode)
        sample *= volume / 100
        wave_list.append(sample)

    if decay_enabled:
        n = len(wave_list)
        for i in range(n):
            wave_list[i] *= math.exp(-i / n)

    return np.array(wave_list, dtype=np.float32)


def tone_square(tone, length, volume, samplerate, bpm, effects=[]):
    distortion_mode = "c"
    decay_enabled = False
    ring_enabled = False

    mapping_distortion = {"dist": "d", "over": "o", "fuzz": "f", "clean": "c"}
    for eff in effects:
        if eff in mapping_distortion:
            distortion_mode = mapping_distortion[eff]
        elif eff == "decay":
            decay_enabled = True
        elif eff == "ring":
            ring_enabled = True

    seconds_per_beat = 60 / bpm
    duration = length * seconds_per_beat
    sample_length = int(samplerate * duration)

    wave_list = []
    freq = convert_tone(tone)
    for i in range(sample_length):
        t = i / samplerate
        sample = 1.0 if math.sin(2 * math.pi * freq * t) >= 0 else -1.0
        sample = all_in_one(sample, distortion_mode)
        sample *= volume / 100
        wave_list.append(sample)

    if decay_enabled:
        n = len(wave_list)
        for i in range(n):
            wave_list[i] *= math.exp(-i / n)

    return np.array(wave_list, dtype=np.float32)


global keywords
keywords = ["bpm", "tsu", "tsl", "vol", "wave", "mes", "clean", "dist", "over", "decay", "ring", "ton", "len"]


def parse_num(string: str):
    if string.isnumeric():
        return int(string)

    else:
        if string[-1] in ".d":
            return int(string[:-1]) + 0.5

    tones = {
        "Ab": -1,
        "A": 0,
        "A#": 1,
        "Bb": 1,
        "B": 2,
        "Cb": 2,
        "B#": 3,
        "C": 3,
        "C#": 4,
        "Db": 4,
        "D": 5,
        "D#": 6,
        "Eb": 6,
        "E": 7,
        "Fb": 7,
        "E#": 8,
        "F": 8,
        "F#": 9,
        "Gb": 9,
        "G": 10,
        "G#": 11,
    }

    raw_tone = re.sub(r"\d+", "", string)
    if raw_tone == "p":
        return "p"

    if raw_tone != "" and raw_tone in list(tones.keys()):
        try:
            octave = int(string.replace(raw_tone, ""))
        except ValueError:
            octave = 4
        semitone = tones[raw_tone] + octave * 12
        return semitone

    return string


def parse_data(data: str):
    global keywods
    out: dict = {}

    out["meta"] = {}

    block = ""

    queue = ""

    menu = ""


    clean_data = re.sub(r"#.*", "", data)

    for i in re.split(r"\s+", clean_data, maxsplit=0, flags=re.DOTALL):
        if len(i) == 0:
            continue

        if i[0] != "[" and i[-1] != ";":
            # Non-blocks, outer
            

            if i == "end":
                block = ""
                queue = ""
                continue

            if i in keywords:
                queue = i

            else:
                path = "meta" if block == "" else block
                if i.isnumeric() or len(i) == 1 or type(parse_num(i)) in [int, float]:
                    if queue not in list(out[path].keys()):
                        if menu == "":
                            out[path][queue] = parse_num(i)
                        else:
                            out[path][menu][queue] = parse_num(i)

                    else:
                        if menu == "":
                            args = out[path][queue]
                        else:
                            args = out[path][menu][queue]

                        if type(args) in [str, int, float]:
                            if menu == "":
                                out[path][queue] = [args, parse_num(i)]
                            else:
                                out[path][menu][queue] = [args, parse_num(i)]

                        elif type(args) == list:
                            args.extend([parse_num(i)])
                            if menu == "":
                                out[path][queue] = args
                            else:
                                out[path][menu][queue] = args

                elif i in keywords:
                    queue = i

        elif i[0] == "[":
            block = i[1:-1]
            queue = ""
            menu = ""

            keywords.append(block)

            if block not in list(out.keys()):
                out[block] = {}

        elif i[-1] == ";":
            if i != "ext;":
                menu = i
                queue = ""
                path = "meta" if block == "" else block
                if menu not in list(out[path].keys()):
                    out[path][menu] = {}
            else:
                menu = ""
                queue = ""

    return out

convert = {"d":"dist", "f":"fuzz", "o":"over", "c":"clean"}

#wave = tone_tri(tone, 4, 50, sample_rate, 120, [convert[dist], "decay"])

#sd.play(wave, samplerate=sample_rate)
#sd.wait()  # Wait until finished


def data_to_soundwave(data: dict) -> npt.NDArray[np.float32]:
    bpm = 120
    tsu = 4
    tsl = 4

    if "meta" in list(data.keys()):
        if "bpm" in list(data["meta"].keys()):
            bpm = data["meta"]["bpm"]

        if "tsu" in list(data["meta"].keys()):
            tsu = data["meta"]["tsu"]

        if "tsl" in list(data["meta"].keys()):
            tsl = data["meta"]["tsl"]
    
    if "def" not in list(data.keys()):
        return np.array([], dtype=np.float32)

    effects = []

    wave = []

    for loop, times in data["def"].items():
        if "fx;" in list(data[loop].keys()):
            effects.extend(list(data[loop]["fx;"].keys()))

        vol = 100
        if "vol" in list(data[loop].keys()):
            vol = data[loop]["vol"]

        wavetype = "sin"
        if "wave" in list(data[loop].keys()):
            wavetype = data[loop]["wave"]

        for _ in range(times):
            tones = data[loop]["ton"]
            lengths = data[loop]["len"]

            for j in range(len(tones)):
                if tones[j] != "p":
                    if lengths[j] == math.floor(lengths[j]):
                        new_length = tsu / lengths[j]

                    else:
                        new_length = tsu / math.floor(lengths[j]) * 1.5

                    args = (tones[j], new_length, vol, sample_rate, bpm, effects)

                    curr_wave = []
                    match wavetype:
                        case "sin":
                            curr_wave = tone_sine(*args)
                        case "squ":
                            curr_wave = tone_square(*args)
                        case "tri":
                            curr_wave = tone_tri(*args)
                        case "saw":
                            curr_wave = tone_saw(*args)

                    wave.extend(curr_wave)
                else:
                    wave.extend([0] * int(sample_rate * 60 / bpm * tsu / lengths[j]))
    
    return np.array(wave, dtype=np.float32)

data = parse_data(open(sys.argv[1]).read())
print(data)

sd.play(data_to_soundwave(data), samplerate=sample_rate)
sd.wait()

