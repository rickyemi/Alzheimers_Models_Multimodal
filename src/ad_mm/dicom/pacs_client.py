"""PACS connectivity: DIMSE (C-ECHO / C-FIND / C-MOVE / C-STORE via pynetdicom) and DICOMweb (QIDO-RS / WADO-RS / STOW-RS).

Typical clinical integration:
    1. Modality -> PACS (Orthanc / vendor PACS) stores T1w MRI.
    2. A routing rule (or worklist poll) sends the study to the AI node (C-STORE to our SCP, or C-MOVE pull).
    3. The AI node de-identifies, runs inference, and pushes back a Secondary Capture / DICOM SR (C-STORE or STOW-RS).

`python -m ad_mm.dicom.pacs_client --demo` starts a local in-process Storage SCP, performs C-ECHO and
C-STORE of five de-identified studies, and reports the round trip (no external PACS required).
For a real PACS use docker-compose (Orthanc on 4242 DIMSE / 8042 DICOMweb) in deployment/docker/.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from pydicom import dcmread
from pynetdicom import AE, evt, AllStoragePresentationContexts, debug_logger  # noqa: F401
from pynetdicom.sop_class import (MRImageStorage, PatientRootQueryRetrieveInformationModelFind,
                                  PatientRootQueryRetrieveInformationModelMove, SecondaryCaptureImageStorage, Verification)

from ad_mm.config import METRIC_DIR, ROOT


class PACSClient:
    def __init__(self, host="localhost", port=4242, called_aet="ORTHANC", calling_aet="AD_MM_AI"):
        self.host, self.port, self.called = host, port, called_aet
        self.ae = AE(ae_title=calling_aet)
        self.ae.add_requested_context(Verification)
        self.ae.add_requested_context(MRImageStorage)
        self.ae.add_requested_context(SecondaryCaptureImageStorage)
        self.ae.add_requested_context(PatientRootQueryRetrieveInformationModelFind)
        self.ae.add_requested_context(PatientRootQueryRetrieveInformationModelMove)

    def _assoc(self):
        a = self.ae.associate(self.host, self.port, ae_title=self.called)
        if not a.is_established:
            raise ConnectionError(f"Association to {self.called}@{self.host}:{self.port} failed")
        return a

    def echo(self) -> int:
        a = self._assoc()
        st = a.send_c_echo()
        a.release()
        return int(st.Status)

    def store(self, files: list[Path]) -> list[int]:
        a = self._assoc()
        out = [int(a.send_c_store(dcmread(f)).Status) for f in files]
        a.release()
        return out

    def find_studies(self, patient_id: str = "*", modality: str = "MR") -> list[dict]:
        from pydicom.dataset import Dataset

        q = Dataset()
        q.QueryRetrieveLevel = "STUDY"
        q.PatientID = patient_id
        q.StudyInstanceUID = ""
        q.StudyDate = ""
        q.ModalitiesInStudy = modality
        a = self._assoc()
        res = [{"PatientID": str(i.PatientID), "StudyInstanceUID": str(i.StudyInstanceUID)}
               for st, i in a.send_c_find(q, PatientRootQueryRetrieveInformationModelFind) if st and st.Status in (0xFF00, 0xFF01)]
        a.release()
        return res

    def move_study(self, study_uid: str, dest_aet: str = "AD_MM_AI") -> int:
        from pydicom.dataset import Dataset

        q = Dataset()
        q.QueryRetrieveLevel = "STUDY"
        q.StudyInstanceUID = study_uid
        a = self._assoc()
        last = None
        for st, _ in a.send_c_move(q, dest_aet, PatientRootQueryRetrieveInformationModelMove):
            last = st
        a.release()
        return int(last.Status) if last else -1


class DICOMwebClient:
    """Minimal DICOMweb client (QIDO-RS search, WADO-RS retrieve, STOW-RS store) using `requests`."""

    def __init__(self, base_url="http://localhost:8042/dicom-web", auth=None):
        import requests

        self.base, self.s = base_url.rstrip("/"), requests.Session()
        if auth:
            self.s.auth = auth

    def search_studies(self, **params) -> list[dict]:
        r = self.s.get(f"{self.base}/studies", params=params, headers={"Accept": "application/dicom+json"})
        r.raise_for_status()
        return r.json()

    def retrieve_instance(self, study, series, instance) -> bytes:
        r = self.s.get(f"{self.base}/studies/{study}/series/{series}/instances/{instance}",
                       headers={"Accept": 'multipart/related; type="application/dicom"'})
        r.raise_for_status()
        return r.content

    def store(self, files: list[Path]) -> int:
        boundary = "AD_MM_BOUNDARY"
        body = b"".join(b"--" + boundary.encode() + b"\r\nContent-Type: application/dicom\r\n\r\n" + Path(f).read_bytes() + b"\r\n"
                        for f in files) + b"--" + boundary.encode() + b"--"
        r = self.s.post(f"{self.base}/studies", data=body,
                        headers={"Content-Type": f'multipart/related; type="application/dicom"; boundary={boundary}'})
        return r.status_code


def start_local_scp(port: int, out_dir: Path):
    """In-process Storage SCP (stands in for the AI node's receiver or a test PACS)."""
    received = []

    def on_store(event):
        ds = event.dataset
        ds.file_meta = event.file_meta
        p = out_dir / f"{ds.SOPInstanceUID}.dcm"
        ds.save_as(p, enforce_file_format=True)
        received.append(p)
        return 0x0000

    ae = AE(ae_title="AD_MM_AI")
    ae.add_supported_context(Verification)
    ae.add_supported_context(MRImageStorage)
    ae.add_supported_context(SecondaryCaptureImageStorage)
    server = ae.start_server(("127.0.0.1", port), block=False, evt_handlers=[(evt.EVT_C_STORE, on_store)])
    return server, received


def demo(port: int = 11112) -> dict:
    files = sorted((ROOT / "data" / "interim" / "dicom_deid").glob("*.dcm"))[:5]
    with tempfile.TemporaryDirectory() as td:
        server, received = start_local_scp(port, Path(td))
        try:
            client = PACSClient("127.0.0.1", port, called_aet="AD_MM_AI", calling_aet="PACS_TEST")
            t0 = time.time()
            echo = client.echo()
            stored = client.store(files)
            dt = time.time() - t0
            time.sleep(0.2)
            result = {"c_echo_status": hex(echo), "c_store_statuses": [hex(s) for s in stored],
                      "instances_received_by_scp": len(received), "round_trip_s": round(dt, 3)}
        finally:
            server.shutdown()
    with open(METRIC_DIR / "pacs_demo.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    print(json.dumps(demo(), indent=2) if args.demo else __doc__)
