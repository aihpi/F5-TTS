import requests

def main():
    # Server URL
    url = 'http://gx12.matthias.herzog.vpn.hpi-sci.de:8005/generate'

    # Paths to your files
    ref_audio_path = '/path/to/ref_voice.wav'  # Replace with your reference audio file path
    generated_audio_path = 'generated_audio.wav'   # The output file

    # Data to send
    data = {
        'ref_text': '',  # Optional reference text
        'text': 'This is the text to generate speech from.',  # The text to synthesize
    }

    # Files to send
    files = {
        'ref_audio': open(ref_audio_path, 'rb'),
    }

    # Send POST request
    response = requests.post(url, data=data, files=files)

    # Check if request was successful
    if response.status_code == 200:
        # Save the audio content to a file
        with open(generated_audio_path, 'wb') as f:
            f.write(response.content)
        print(f'Generated audio saved as {generated_audio_path}')
    else:
        print('Error:', response.status_code, response.text)

if __name__ == '__main__':
    main()
