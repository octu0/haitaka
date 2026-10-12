import XCTest
@testable import Haitaka

final class VADTests: XCTestCase {
    
    func testSilenceDetection() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 320)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024)
        
        let silenceFrame = [Float](repeating: 0.0, count: 320)
        let result = silenceFrame.withUnsafeBufferPointer { ptr in
            return vad.processFrame(ptr: ptr.baseAddress!, count: 320, workspace: workspace)
        }
        
        XCTAssertFalse(result.isSpeech)
        XCTAssertLessThan(result.rms, 1e-5)
    }
    
    func testPureToneSpeechDetection() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 320)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024)
        
        // 400 Hz sine, amplitude 0.5
        var toneFrame = [Float](repeating: 0.0, count: 320)
        let factor = (2.0 * Float.pi * 400.0) / 16000.0
        var i = 0
        while i < 320 {
            toneFrame[i] = 0.5 * sin(Float(i) * factor)
            i += 1
        }
        
        let result = toneFrame.withUnsafeBufferPointer { ptr in
            return vad.processFrame(ptr: ptr.baseAddress!, count: 320, workspace: workspace)
        }
        
        XCTAssertTrue(result.isSpeech)
        XCTAssertLessThan(0.1, result.rms)
        XCTAssertLessThan(0.5, result.voicingRatio)
    }
    
    func testNoiseFloorTracking() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 320)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024)
        
        // Low-level noise frames (amplitude 0.001)
        var noiseFrame = [Float](repeating: 0.0, count: 320)
        var i = 0
        while i < 320 {
            noiseFrame[i] = Float(((i % 7) - 3)) * 0.0003
            i += 1
        }
        
        var iter = 0
        var lastNoiseFloor: Float = 0.0
        while iter < 20 {
            let res = noiseFrame.withUnsafeBufferPointer { ptr in
                return vad.processFrame(ptr: ptr.baseAddress!, count: 320, workspace: workspace)
            }
            lastNoiseFloor = res.noiseFloor
            iter += 1
        }
        
        // The noise floor estimate has been updated to a positive value.
        XCTAssertLessThan(0.0, lastNoiseFloor)
    }
    
    func testVoicingRatioDiscrimination() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 320)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024)
        
        // Voiced frame (200 Hz)
        var voicedFrame = [Float](repeating: 0.0, count: 320)
        let factor = (2.0 * Float.pi * 200.0) / 16000.0
        var i = 0
        while i < 320 {
            voicedFrame[i] = 0.5 * sin(Float(i) * factor)
            i += 1
        }
        
        let voicedRes = voicedFrame.withUnsafeBufferPointer { ptr in
            return vad.processFrame(ptr: ptr.baseAddress!, count: 320, workspace: workspace)
        }
        
        // Pseudo white noise frame
        var noiseFrame = [Float](repeating: 0.0, count: 320)
        i = 0
        while i < 320 {
            let pseudoRand = Float((i * 1103515245 + 12345) & 0x7FFFFFFF) / Float(0x7FFFFFFF)
            noiseFrame[i] = (pseudoRand - 0.5) * 0.02
            i += 1
        }
        
        let noiseRes = noiseFrame.withUnsafeBufferPointer { ptr in
            return vad.processFrame(ptr: ptr.baseAddress!, count: 320, workspace: workspace)
        }
        
        XCTAssertLessThan(noiseRes.voicingRatio, voicedRes.voicingRatio)
        XCTAssertLessThan(0.6, voicedRes.voicingRatio)
    }
    
    func testUtteranceSegmentationPrePostRoll() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 320, hopSize: 160)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024)
        
        let totalLength = 16000 * 2 // 2 s
        var audio = [Float](repeating: 0.0, count: totalLength)
        
        // 300 Hz tone from 0.5 s to 1.2 s
        let speechStart = 8000
        let speechEnd = 19200
        let factor = (2.0 * Float.pi * 300.0) / 16000.0
        
        var i = speechStart
        while i < speechEnd {
            audio[i] = 0.6 * sin(Float(i - speechStart) * factor)
            i += 1
        }
        
        let segments = vad.segmentUtterances(pcmData: audio, workspace: workspace)
        
        XCTAssertEqual(segments.count, 1)
        if segments.count == 1 {
            let seg = segments[0]
            // Pre-roll starts the segment before the tone onset.
            XCTAssertLessThanOrEqual(seg.startIndex, speechStart)
            // Post-roll (hangover) ends the segment after the tone offset.
            XCTAssertLessThanOrEqual(speechEnd, seg.endIndex)
            XCTAssertLessThan(0.7, seg.durationSeconds)
        }
    }
    
    func testFricativeConsonantDetection() {
        let config = DSPConfig(sampleRate: 16000, frameSize: 320)
        let vad = VAD(config: config)
        let workspace = DSPWorkspace(maxFrameSize: 1024)
        
        // Alternating waveform with high ZCR and moderate energy
        var fricativeFrame = [Float](repeating: 0.0, count: 320)
        var i = 0
        while i < 320 {
            var sign: Float = -1.0
            if i % 2 == 0 {
                sign = 1.0
            }
            fricativeFrame[i] = sign * 0.05
            i += 1
        }
        
        let result = fricativeFrame.withUnsafeBufferPointer { ptr in
            return vad.processFrame(ptr: ptr.baseAddress!, count: 320, workspace: workspace)
        }
        
        XCTAssertLessThan(0.35, result.zcr)
        XCTAssertLessThan(0.005, result.rms)
    }
}
