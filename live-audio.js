document.addEventListener('DOMContentLoaded', function() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    let voiceChart;

    // Initialize chart (smaller pie chart with more emotions)
    function initializeChart() {
        voiceChart = new Chart(document.getElementById('voiceChart').getContext('2d'), {
            type: 'pie',
            data: { 
                labels: ['Confident', 'Neutral', 'Fear', 'More Fear', 'Hesitant'], 
                datasets: [{ 
                    data: [20, 20, 20, 20, 20], // Initial equal distribution
                    backgroundColor: ['#22d3ee', '#6366f1', '#f87171', '#dc2626', '#f43f5e'] 
                }] 
            },
            options: { 
                responsive: true,
                maintainAspectRatio: false,
                plugins: { 
                    legend: { position: 'bottom', labels: { font: { size: 12 } } }
                },
                layout: {
                    padding: 20
                }
            }
        });
    }
    initializeChart();

    // Speech recognition
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = function(event) {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript + ' ';
        }
        document.getElementById('transcriptText').textContent = transcript.trim();
        document.getElementById('correctedTranscript').textContent = correctTranscript(transcript.trim());
        analyzeVoiceTone(transcript.trim());
    };

    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
    };

    function correctTranscript(transcript) {
        if (!transcript) return 'No transcription available';
        let corrected = transcript
            .replace(/\b(um|uh|like|you know|basically|actually|literally|sort of|kind of)\b/gi, '')
            .replace(/\b(\w+)\s+\1\b/gi, '$1')
            .replace(/\s+/g, ' ')
            .trim()
            .replace(/(?:^|\.\s+)([a-z])/g, match => match.toUpperCase())
            .replace(/\b[i]\b/g, 'I')
            .replace(/([.!?])\s*([a-z])/g, (match, p1, p2) => p1 + ' ' + p2.toUpperCase());
        return corrected || 'No transcription available';
    }

    function analyzeVoiceTone(transcript) {
        if (!transcript) {
            voiceChart.data.datasets[0].data = [20, 20, 20, 20, 20]; // Default if no speech
            voiceChart.update();
            return;
        }

        const words = transcript.toLowerCase().split(' ').filter(word => word.length > 0);
        const confidenceWords = ['definitely', 'absolutely', 'certainly', 'confident', 'sure', 'certain'];
        const hesitantWords = ['maybe', 'perhaps', 'possibly', 'think', 'guess', 'unsure'];
        const fearWords = ['scared', 'afraid', 'nervous', 'terrified', 'worried'];
        const moreFearWords = ['petrified', 'panicked', 'horrified', 'terrifyingly', 'paralyzed'];

        let confident = 0, hesitant = 0, fear = 0, moreFear = 0, total = words.length || 1;

        words.forEach(word => {
            if (confidenceWords.includes(word)) confident++;
            if (hesitantWords.includes(word)) hesitant++;
            if (fearWords.includes(word)) fear++;
            if (moreFearWords.includes(word)) moreFear++;
        });

        // Dynamic weights based on word frequency and intensity
        const confidentWeight = Math.min(0.7, (confident / total) * 1.5);
        const hesitantWeight = Math.min(0.5, (hesitant / total) * 1.2);
        const fearWeight = Math.min(0.4, (fear / total) * 1.3);
        const moreFearWeight = Math.min(0.3, (moreFear / total) * 1.5);
        const neutralWeight = Math.max(0.1, 1 - (confidentWeight + hesitantWeight + fearWeight + moreFearWeight));

        const toneAnalysis = {
            confident: confidentWeight,
            neutral: neutralWeight,
            fear: fearWeight,
            moreFear: moreFearWeight,
            hesitant: hesitantWeight
        };

        // Normalize to sum to 100% for pie chart
        const sum = toneAnalysis.confident + toneAnalysis.neutral + toneAnalysis.fear + toneAnalysis.moreFear + toneAnalysis.hesitant;
        voiceChart.data.datasets[0].data = [
            (toneAnalysis.confident / sum) * 100,
            (toneAnalysis.neutral / sum) * 100,
            (toneAnalysis.fear / sum) * 100,
            (toneAnalysis.moreFear / sum) * 100,
            (toneAnalysis.hesitant / sum) * 100
        ];
        voiceChart.update();
    }

    startBtn.addEventListener('click', function() {
        recognition.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
    });

    stopBtn.addEventListener('click', function() {
        recognition.stop();
        startBtn.disabled = false;
        stopBtn.disabled = true;
        document.getElementById('transcriptText').textContent = 'Transcript will appear here...';
        document.getElementById('correctedTranscript').textContent = 'Corrected transcript will appear here...';
        voiceChart.data.datasets[0].data = [20, 20, 20, 20, 20]; // Reset chart
        voiceChart.update();
    });
});