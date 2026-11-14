"""Testy check_acts_freshness.check_act: rozróżnienie PROBLEMÓW (psują wynik)
od INFORMACJI (sama zmiana `changeDate`).
Odpowiedzi ELI API są podstawiane -- bez sieci."""

import check_acts_freshness as caf


def _meta(file_name="D1Lj.pdf", change="2026-10-01T00:00:00", status="obowiązujący", in_force="IN_FORCE", repealing=None):
    m = {
        "status": status,
        "inForce": in_force,
        "changeDate": change,
        "texts": [{"type": "U", "fileName": file_name}],
        "references": {},
    }
    if repealing:
        m["references"]["Akty uchylające"] = [{"id": r} for r in repealing]
    return m


ACT = {"short": "x", "publisher": "DU", "year": 2000, "position": 1, "domains": ["d"]}


def _run(monkeypatch, local, remote, act=ACT):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return remote

    monkeypatch.setattr(caf, "load_local_meta", lambda short: local)
    monkeypatch.setattr(caf.requests, "get", lambda *a, **k: Resp())
    return caf.check_act(act)


def test_identical_meta_is_ok(monkeypatch):
    assert _run(monkeypatch, _meta(), _meta()) == ([], [])


def test_only_change_date_is_info_not_problem(monkeypatch):
    problems, infos = _run(monkeypatch, _meta(), _meta(change="2026-11-01T00:00:00"))
    assert problems == []
    assert len(infos) == 1 and "metadane" in infos[0]


def test_new_consolidated_text_is_a_problem(monkeypatch):
    problems, _ = _run(monkeypatch, _meta(), _meta(file_name="D2Lj.pdf"))
    assert any("nowszy tekst" in p for p in problems)


def test_repealed_act_is_a_problem(monkeypatch):
    remote = _meta(status="uchylony", in_force="NOT_IN_FORCE", repealing=["DU/2024/1"])
    problems, _ = _run(monkeypatch, _meta(), remote)
    assert any("nie wygląda na obowiązujący" in p for p in problems)
    assert any("UCHYLONY" in p for p in problems)


def test_missing_local_meta_is_a_problem(monkeypatch):
    monkeypatch.setattr(caf, "load_local_meta", lambda short: None)
    problems, infos = caf.check_act(ACT)
    assert problems and not infos
