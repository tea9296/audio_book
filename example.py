from helper.inf import generate_audio, generate_book

text = "Just a weekly reminder for those who wish to contribute to the project I'm working on, love you!"
ref_audio = ["F:/project/fish-audio/preprocess_data/fauna/0000.wav"]
ref_text = "I don't really see the need for a love triangle. You hate the triangle. I don't like it either. I just don't see the need."
output_path = "F:/project/audio_book/output/fauna01"
# generate_audio(text=text,
#                reference_audio=ref_audio,
#                reference_text=ref_text,
#                output=output_path,
#                streaming=True,
#                temperature=0.1)

generate_book()
