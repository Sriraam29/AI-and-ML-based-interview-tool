document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('dropZone');
    const videoInput = document.getElementById('videoInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    let faceChart, voiceChart;
    let faceDetector, faceLandmarksDetector;
    let selectedVideo = null;
    let isAnalyzing = false;

    // Initialize models
    async function initializeModels() {
        try {
            await tfjs.ready();
            faceDetector = await faceDetection.createDetector(faceDetection.SupportedModels.MediaPipeFaceDetector, { runtime: 'tfjs' });
            faceLandmarksDetector = await faceLandmarksDetection.createDetector(faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, { runtime: 'tfjs' });
        } catch (error) {
            console.error('Error loading models:', error);
        }
    }
    initializeModels();

    // Initialize charts (smaller pie charts)
    function initializeCharts() {
        faceChart = new Chart(document.getElementById('faceChart').getContext('2d'), {
            type: 'pie',
            data: { labels: ['Happy', 'Neutral', 'Anxious', 'Other'], datasets: [{ data: [25, 25, 25, 25], backgroundColor: ['#4ade80', '#6366f1', '#f43f5e', '#94a3b8'] }] },
            options: { 
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { font: { size: 12 } } } },
                layout: { padding: 20 }
            }
        });
        voiceChart = new Chart(document.getElementById('voiceChart').getContext('2d'), {
            type: 'pie',
            data: { labels: ['Confident', 'Neutral', 'Hesitant', 'Other'], datasets: [{ data: [25, 25, 25, 25], backgroundColor: ['#4ade80', '#6366f1', '#f43f5e', '#94a3b8'] }] },
            options: { 
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { font: { size: 12 } } } },
                layout: { padding: 20 }
            }
        });
    }
    initializeCharts();

    // Process video frame
    async function processVideoFrame(video) {
        try {
            const faces = await faceDetector.estimateFaces(video);
            if (faces.length > 0) {
                const landmarks = await faceLandmarksDetector.estimateFaces(video);
                return analyzeFacialExpression(landmarks[0]);
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        }
        return null;
    }

    function analyzeFacialExpression(landmarks) {
        const expressions = { happy: 0.35, neutral: 0.40, anxious: 0.15, other: 0.10 };
        const sum = expressions.happy + expressions.neutral + expressions.anxious + expressions.other;
        faceChart.data.datasets[0].data = [
            (expressions.happy / sum) * 100,
            (expressions.neutral / sum) * 100,
            (expressions.anxious / sum) * 100,
            (expressions.other / sum) * 100
        ];
        faceChart.update();
        return expressions;
    }

    function updateVoiceChart(toneAnalysis) {
        const sum = toneAnalysis.confident + toneAnalysis.neutral + toneAnalysis.hesitant + toneAnalysis.other;
        voiceChart.data.datasets[0].data = [
            (toneAnalysis.confident / sum) * 100,
            (toneAnalysis.neutral / sum) * 100,
            (toneAnalysis.hesitant / sum) * 100,
            (toneAnalysis.other / sum) * 100
        ];
        voiceChart.update();
    }

    function analyzeVoiceTone(transcript) {
        if (!transcript) {
            voiceChart.data.datasets[0].data = [25, 25, 25, 25];
            voiceChart.update();
            return;
        }

        const words = transcript.toLowerCase().split(' ').filter(word => word.length > 0);
        const confidenceWords = ['definitely', 'absolutely', 'certainly', 'confident', 'sure'];
        const hesitantWords = ['maybe', 'perhaps', 'possibly', 'think', 'guess'];
        let confident = 0, hesitant = 0, total = words.length || 1;

        words.forEach(word => {
            if (confidenceWords.includes(word)) confident++;
            if (hesitantWords.includes(word)) hesitant++;
        });

        const confidentWeight = Math.min(0.7, (confident / total) * 1.5);
        const hesitantWeight = Math.min(0.5, (hesitant / total) * 1.2);
        const neutralWeight = Math.max(0.1, 1 - (confidentWeight + hesitantWeight));
        const otherWeight = Math.max(0.1, 1 - (confidentWeight + hesitantWeight + neutralWeight));

        const toneAnalysis = {
            confident: confidentWeight,
            neutral: neutralWeight,
            hesitant: hesitantWeight,
            other: otherWeight
        };

        updateVoiceChart(toneAnalysis);
    }

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

    // Video handling with Web Audio API (simulated for now)
    async function handleVideoAnalysis() {
        if (!selectedVideo || isAnalyzing) return;
        isAnalyzing = true;
        analyzeBtn.disabled = true;

        const video = document.createElement('video');
        video.src = URL.createObjectURL(selectedVideo);
        video.muted = false;

        try {
            await new Promise((resolve, reject) => {
                video.onloadedmetadata = () => {
                    document.getElementById('videoDuration').textContent = 
                        `${Math.floor(video.duration / 60)}:${(video.duration % 60).toString().padStart(2, '0')}`;
                    resolve();
                };
                video.onerror = () => reject(new Error('Failed to load video'));
            });

            video.play();

            const processInterval = setInterval(async () => {
                if (video.paused || video.ended) {
                    clearInterval(processInterval);
                    isAnalyzing = false;
                    analyzeBtn.disabled = false;
                    URL.revokeObjectURL(video.src);

                    // Simulate transcription after video ends (requires external API for real implementation)
                    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    const source = audioContext.createMediaElementSource(video);
                    source.connect(audioContext.destination); // Just for playback

                    // Placeholder: Simulate transcript after video ends
                    const simulatedTranscript = "This is a sample interview. I am definitely confident about my skills.";
                    document.getElementById('transcriptText').textContent = simulatedTranscript;
                    const corrected = correctTranscript(simulatedTranscript);
                    document.getElementById('correctedTranscript').textContent = corrected;
                    analyzeVoiceTone(simulatedTranscript);
                    audioContext.close();
                    return;
                }
                await processVideoFrame(video);
            }, 100);

        } catch (error) {
            console.error('Error analyzing video:', error);
            isAnalyzing = false;
            analyzeBtn.disabled = false;
            URL.revokeObjectURL(video.src);
        }
    }

    function handleFileSelection(file) {
        if (!file || !file.type.startsWith('video/')) {
            alert('Please select a valid video file.');
            return;
        }
        selectedVideo = file;
        analyzeBtn.disabled = false;
    }

    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFileSelection(e.target.files[0]);
    });

    analyzeBtn.addEventListener('click', handleVideoAnalysis);

    dropZone.addEventListener('dragover', (e) => e.preventDefault());
    dropZone.addEventListener('dragleave', (e) => e.preventDefault());
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length > 0) handleFileSelection(e.dataTransfer.files[0]);
    });
});