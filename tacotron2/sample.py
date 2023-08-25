from TTS.api import TTS

# Running a multi-speaker and multi-lingual model

# List available 🐸TTS models and choose the first one
model_name = TTS.list_models()[1]
# Init TTS
tts = TTS(model_name)

# Run TTS

# ❗ Since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# Text to speech with a numpy output
wav = tts.tts("We will rock you", speaker=tts.speakers[0], language=tts.languages[0])
# Text to speech to a file

tts.tts_to_file(text="We will rock you", language=tts.languages[0], speaker=tts.speakers[0], file_path='sound.wav')

['tts_models/multilingual/multi-dataset/your_tts', 'tts_models/multilingual/multi-dataset/bark', 'tts_models/bg/cv/vits', 'tts_models/cs/cv/vits', 'tts_models/da/cv/vits', 'tts_models/et/cv/vits', 'tts_models/ga/cv/vits', 'tts_models/en/ek1/tacotron2', 'tts_models/en/ljspeech/tacotron2-DDC', 'tts_models/en/ljspeech/tacotron2-DDC_ph', 'tts_models/en/ljspeech/glow-tts', 'tts_models/en/ljspeech/speedy-speech', 'tts_models/en/ljspeech/tacotron2-DCA', 'tts_models/en/ljspeech/vits', 'tts_models/en/ljspeech/vits--neon', 'tts_models/en/ljspeech/fast_pitch', 'tts_models/en/ljspeech/overflow', 'tts_models/en/ljspeech/neural_hmm', 'tts_models/en/vctk/vits', 'tts_models/en/vctk/fast_pitch', 'tts_models/en/sam/tacotron-DDC', 'tts_models/en/blizzard2013/capacitron-t2-c50', 'tts_models/en/blizzard2013/capacitron-t2-c150_v2', 'tts_models/en/multi-dataset/tortoise-v2', 'tts_models/en/jenny/jenny', 'tts_models/es/mai/tacotron2-DDC', 'tts_models/es/css10/vits', 'tts_models/fr/mai/tacotron2-DDC', 'tts_models/fr/css10/vits', 'tts_models/uk/mai/glow-tts', 'tts_models/uk/mai/vits', 'tts_models/zh-CN/baker/tacotron2-DDC-GST', 'tts_models/nl/mai/tacotron2-DDC', 'tts_models/nl/css10/vits', 'tts_models/de/thorsten/tacotron2-DCA', 'tts_models/de/thorsten/vits', 'tts_models/de/thorsten/tacotron2-DDC', 'tts_models/de/css10/vits-neon', 'tts_models/ja/kokoro/tacotron2-DDC', 'tts_models/tr/common-voice/glow-tts', 'tts_models/it/mai_female/glow-tts', 'tts_models/it/mai_female/vits', 'tts_models/it/mai_male/glow-tts', 'tts_models/it/mai_male/vits', 'tts_models/ewe/openbible/vits', 'tts_models/hau/openbible/vits', 'tts_models/lin/openbible/vits', 'tts_models/tw_akuapem/openbible/vits', 'tts_models/tw_asante/openbible/vits', 'tts_models/yor/openbible/vits', 'tts_models/hu/css10/vits', 'tts_models/el/cv/vits', 'tts_models/fi/css10/vits', 'tts_models/hr/cv/vits', 'tts_models/lt/cv/vits', 'tts_models/lv/cv/vits', 'tts_models/mt/cv/vits', 'tts_models/pl/mai_female/vits', 'tts_models/pt/cv/vits', 'tts_models/ro/cv/vits', 'tts_models/sk/cv/vits', 'tts_models/sl/cv/vits', 'tts_models/sv/cv/vits', 'tts_models/ca/custom/vits', 'tts_models/fa/custom/glow-tts', 'tts_models/bn/custom/vits-male', 'tts_models/bn/custom/vits-female']



# Running a single speaker model

# Init TTS with the target model name
tts = TTS(model_name="tts_models/en/ljspeech/vits", gpu=False)
# Run TTS
tts.tts_to_file(text="We will rock you", language=tts.languages[0], speaker=tts.speakers[0], file_path='sound.wav')

# Example voice cloning with YourTTS in English, French and Portuguese

tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)
tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="output.wav")
tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr-fr", file_path="output.wav")
tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt-br", file_path="output.wav")


# Example voice conversion converting speaker of the `source_wav` to the speaker of the `target_wav`

tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False, gpu=True)
tts.voice_conversion_to_file(source_wav="my/source.wav", target_wav="my/target.wav", file_path="output.wav")

# Example voice cloning by a single speaker TTS model combining with the voice conversion model. This way, you can
# clone voices by using any model in 🐸TTS.

tts = TTS("tts_models/de/thorsten/tacotron2-DDC")
tts.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)

# Example text to speech using [🐸Coqui Studio](https://coqui.ai) models.

# You can use all of your available speakers in the studio.
# [🐸Coqui Studio](https://coqui.ai) API token is required. You can get it from the [account page](https://coqui.ai/account).
# You should set the `COQUI_STUDIO_TOKEN` environment variable to use the API token.

# If you have a valid API token set you will see the studio speakers as separate models in the list.
# The name format is coqui_studio/en/<studio_speaker_name>/coqui_studio
models = TTS().list_models()
# Init TTS with the target studio speaker
tts = TTS(model_name="coqui_studio/en/Torcull Diarmuid/coqui_studio", progress_bar=False, gpu=False)
# Run TTS
tts.tts_to_file(text="This is a test.", file_path=OUTPUT_PATH)
# Run TTS with emotion and speed control
tts.tts_to_file(text="This is a test.", file_path=OUTPUT_PATH, emotion="Happy", speed=1.5)


#Example text to speech using **Fairseq models in ~1100 languages** 🤯.

#For these models use the following name format: `tts_models/<lang-iso_code>/fairseq/vits`.
#You can find the list of language ISO codes [here](https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html) and learn about the Fairseq models [here](https://github.com/facebookresearch/fairseq/tree/main/examples/mms).

# TTS with on the fly voice conversion
api = TTS("tts_models/deu/fairseq/vits")
api.tts_with_vc_to_file(
    "Wie sage ich auf Italienisch, dass ich dich liebe?",
    speaker_wav="target/speaker.wav",
    file_path="output.wav"
)