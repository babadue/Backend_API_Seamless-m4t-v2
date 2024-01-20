# text_translate_api.py
from model_initializer import initialize_model
from flask import Flask, request, jsonify
import sounddevice as sd
from flask_cors import CORS  # Import CORS from flask_cors


model, processor, device = initialize_model()
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in your Flask app

@app.route('/t2t', methods=['POST'])
def t2t():
    input_text = request.json.get('inputText', '')
    srcLang = request.json.get('srcLang', '')
    tgtLang = request.json.get('tgtLang', '')
    print(f"t2t input_text: {input_text}, srcLang: {srcLang}, tgtLang: {tgtLang}")
    # process input
    text_inputs = processor(text=f"{input_text}", src_lang=srcLang, return_tensors="pt").to(device)

    # generate translation
    output_tokens = model.generate(**text_inputs, tgt_lang=tgtLang, generate_speech=False)
    translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

    return jsonify({'processedText': translated_text_from_text})
    
@app.route('/t2s', methods=['POST'])
def t2s():
    input_text = request.json.get('inputText', '')
    srcLang = request.json.get('srcLang', '')
    tgtLang = request.json.get('tgtLang', '')
    print(f"t2t input_text: {input_text}, srcLang: {srcLang}, tgtLang: {tgtLang}")
    # process input
    text_inputs = processor(text=f"{input_text}", src_lang=srcLang, return_tensors="pt").to(device)

    # generate translation audio
    audio_array_from_text = model.generate(**text_inputs, tgt_lang=tgtLang)[0].cpu().numpy().squeeze()
    # Set the sample rate
    sample_rate = model.config.sampling_rate
    print(f"sample_rate: {sample_rate}")

    return jsonify({'audioData': audio_array_from_text.tolist(), 'sample_rate': sample_rate})
    
@app.route('/s2t', methods=['POST'])
def s2t():
    audio_sample = request.json.get('audioSample', '')
    sample_rate = request.json.get('sampleRate', '')
    srcLang = request.json.get('srcLang', '')
    tgtLang = request.json.get('tgtLang', '')
    print(f"s2t sample_rate: {sample_rate} srcLang: {srcLang}, tgtLang: {tgtLang}")
    # sd.play(audio_sample, sample_rate)
    # sd.wait()  # Wait for playback to finish
    audio_inputs = processor(audios=audio_sample, sampling_rate=sample_rate, return_tensors="pt").to(device)
    print(f"s2t after audio_inputs")
    output_tokens = model.generate(**audio_inputs, tgt_lang=tgtLang, generate_speech=False)
    print(f"s2t after output_tokens")
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    print(f"s2t after translated_text_from_audio: {translated_text_from_audio}")
    return jsonify({'processedText': translated_text_from_audio})

@app.route('/s2s', methods=['POST'])
def s2s():
    audio_sample = request.json.get('audioSample', '')
    sample_rate = request.json.get('sampleRate', '')
    srcLang = request.json.get('srcLang', '')
    tgtLang = request.json.get('tgtLang', '')
    print(f"s2s sample_rate: {sample_rate} srcLang: {srcLang}, tgtLang: {tgtLang}")
    # sd.play(audio_sample, sample_rate)
    # sd.wait()  # Wait for playback to finish
    audio_inputs = processor(audios=audio_sample, sampling_rate=sample_rate, return_tensors="pt").to(device)
    print(f"s22 after audio_inputs")
    audio_array_from_audio = model.generate(**audio_inputs, tgt_lang=tgtLang)[0].cpu().numpy().squeeze()
    # Set the sample rate
    sample_rate = model.config.sampling_rate
    print(f"sample_rate: {sample_rate}")

    return jsonify({'audioData': audio_array_from_audio.tolist(), 'sample_rate': sample_rate})


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0', port=4000) 
