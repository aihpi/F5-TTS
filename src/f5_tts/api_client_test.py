import os
from pathlib import Path
import requests
from zipfile import ZipFile
import io
import argparse


def simple():
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


class TTSClient:
    def __init__(self, url="http://localhost:8000"):
        self.url = url

    def generate_predefined_scenarios(self, ref_audio_path: str, output_dir: str) -> bool:
        """
        Generate audio for predefined scenarios using the TTS server

        Args:
            ref_audio_path: Path to reference audio file
            output_dir: Directory to save generated audio files

        Returns:
            bool: True if successful, False otherwise
        """
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare the request
        files = {
            'ref_audio': ('ref_audio.mp3', open(ref_audio_path, 'rb'), 'audio/mpeg'),
        }

        data = {
            'ref_text': ''  # Optional reference text
        }

        try:
            # Send request to server
            print(f"Sending request to {self.url}/generate_predefined")
            response = requests.post(
                f"{self.url}/generate_predefined",
                files=files,
                data=data,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()

            # Process the response
            print("Processing response...")
            zip_buffer = io.BytesIO(response.content)

            # Extract files
            with ZipFile(zip_buffer) as zip_file:
                # Get list of files for reporting
                file_list = zip_file.namelist()

                # Extract all files
                zip_file.extractall(output_dir)

            print(f"\nSuccessfully generated {len(file_list)} audio files:")
            for file in sorted(file_list):
                print(f"- {file}")
            print(f"\nFiles saved in: {output_dir}")

            return True

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with server: {str(e)}")
            return False
        except Exception as e:
            print(f"Error processing audio files: {str(e)}")
            return False
        finally:
            files['ref_audio'][1].close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate TTS audio for predefined scenarios')
    parser.add_argument('--ref-audio', type=str, default='ref.mp3',
                        help='Path to reference audio file (default: ref.mp3)')
    parser.add_argument('--output-dir', type=str, default='out/315000',
                        help='Output directory for generated audio (default: out/315000)')
    parser.add_argument('--server-url', type=str, default='http://localhost:8000',
                        help='TTS server URL (default: http://localhost:8000)')

    args = parser.parse_args()

    # Check if reference audio exists
    if not os.path.exists(args.ref_audio):
        print(f"Error: Reference audio file '{args.ref_audio}' not found!")
        return

    # Initialize client and generate audio
    client = TTSClient(args.server_url)
    client.generate_predefined_scenarios(args.ref_audio, args.output_dir)


if __name__ == '__main__':
    main()
