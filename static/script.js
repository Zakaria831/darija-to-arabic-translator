document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('inputText');
    const translateButton = document.getElementById('translateButton');
    const translatedText = document.getElementById('translatedText');
    const voiceButton = document.getElementById('voiceButton');

    // Translate button
    translateButton.addEventListener('click', async () => {
        const userInput = inputText.value.trim();
        if (!userInput) {
            translatedText.innerText = "Please provide text to translate.";
            return;
        }
        await sendToTranslate(userInput);
    });

    // Voice input with Web Speech API
    if ('webkitSpeechRecognition' in window) {
        const recognition = new webkitSpeechRecognition();
        recognition.lang = 'ar-DZ'; 
        recognition.continuous = false;
        recognition.interimResults = false;

        voiceButton.addEventListener('click', () => {
            recognition.start();
        });

        recognition.onresult = (event) => {
            const speechResult = event.results[0][0].transcript;
            inputText.value = speechResult;
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            alert('Voice input error: ' + event.error);
        };
    } else {
        // If not supported, hide the voice button
        voiceButton.style.display = 'none';
    }
});

async function sendToTranslate(text) {
    try {
        const response = await fetch('/translate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text }),
        });

        if (response.ok) {
            const data = await response.json();
            if (data.translation) {
                document.getElementById('translatedText').innerText = data.translation;
            } else if (data.error) {
                document.getElementById('translatedText').innerText = "Error: " + data.error;
            }
        } else {
            document.getElementById('translatedText').innerText = "Error in translation. Please try again.";
        }
    } catch (error) {
        console.error("Error:", error);
        document.getElementById('translatedText').innerText = "An error occurred. Please try again.";
    }
}
