import * as ort from "onnxruntime-node"

export class Model {
    model: ort.InferenceSession
    constructor(modelPath: string) {
        this.model = new ort.InferenceSession(modelPath)
    }
}