# IRIS Classification Deployment (Partial Implementation)

## Overview
This repository is a partial implementation of the CI/CD scaling assignment for the IRIS classification model.  
Objective: deploy an ML inference service to Google Kubernetes Engine (GKE) and demonstrate autoscaling under load.

---

## Completed Work
1. **Containerization**
   - Created Docker image for the IRIS inference API (`iris_app.py`).
   - Pushed image to Google Artifact Registry using:
     ```bash
     gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/ml-apps/iris-service:latest .
     ```

2. **GKE Cluster Setup**
   - Created a single-node cluster:
     ```bash
     gcloud container clusters create gke-iris-cluster \
       --zone us-central1-a \
       --num-nodes=1 \
       --machine-type=e2-standard-2 \
       --disk-size=50
     ```
   - Verified access with:
     ```bash
     kubectl get nodes
     ```

3. **Deployment and Service**
   - Applied deployment and service YAMLs:
     ```bash
     kubectl apply -f deployment.yaml
     kubectl apply -f service.yaml
     ```
   - Service was exposed via `LoadBalancer` or port-forwarding for internal testing.

4. **Horizontal Pod Autoscaler (HPA) Setup**
   - Created or planned HPA configuration:
     ```bash
     kubectl autoscale deployment iris-deployment --cpu-percent=50 --min=1 --max=3
     ```
   - Metrics server enabled for CPU-based scaling.

---

## Pending / Partially Completed
- Stress testing using `wrk` to simulate >1000 concurrent requests.
- Observing automatic scaling (1–3 pods) under heavy load.
- Measuring bottlenecks when scaling limited to 1 pod.
- Collecting detailed performance metrics and logs for comparison.

---

## Files Used
- `iris_app.py` — FastAPI/Flask app serving predictions.  
- `create_model.py` — model preparation logic.  
- `Dockerfile` — container image build definition.  
- `deployment.yaml` — Kubernetes deployment spec.  
- `service.yaml` — service definition for exposure.  
- `hpa.yaml` — autoscaling policy (planned).  
- `payload.lua` — wrk load test script.  
- `requirements.txt` — Python dependencies.

---

## Next Steps (for completion)
1. Run wrk load test:
   ```bash
   wrk -t12 -c1000 -d60 -s payload.lua http://<EXTERNAL_IP>:8000/predict
Observe scaling via:

```bash
kubectl get hpa
kubectl get pods -w
```
Limit scaling (max=1) and repeat with concurrency=2000.

Record throughput, latency, and errors for analysis.