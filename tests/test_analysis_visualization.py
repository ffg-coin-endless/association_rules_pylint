import pandas as pd
from src import visualization

def make_dummy_rules_csv(path):
    """Helper to create a small dummy rules CSV with at least one shared error rule."""
    data = [
        {
            "antecedents": "{'A'}",  # simple set string
            "consequents": "{'E1234'}",  # valid error code pattern
            "Left_Hand_Side": "A",
            "Right_Hand_Side": "E1234",
            "support": 0.5,
            "confidence": 0.8,
            "lift": 1.2,
            "antecedent support": 0.6,
            "consequent support": 0.7,
        },
        {
            "antecedents": "{'E0001'}",
            "consequents": "{'E0002'}",
            "Left_Hand_Side": "E0001",
            "Right_Hand_Side": "E0002",
            "support": 0.3,
            "confidence": 0.7,
            "lift": 2.0,
            "antecedent support": 0.4,
            "consequent support": 0.5,
        },
        {
            "antecedents": "{'E0003'}",
            "consequents": "{'W0001'}",
            "Left_Hand_Side": "E0003",
            "Right_Hand_Side": "W0001",
            "support": 0.2,
            "confidence": 0.6,
            "lift": 1.5,
            "antecedent support": 0.3,
            "consequent support": 0.4,
        },
    ]
    pd.DataFrame(data).to_csv(path, index=False)


def test_plot_grouped_rule_matrix(tmp_path):
    repo_name = "DummyRepo"
    csv_file = tmp_path / f"rules_{repo_name}.csv"
    make_dummy_rules_csv(csv_file)

    out_file = visualization.plot_grouped_rule_matrix(repo_name, in_dir=tmp_path, out_dir=tmp_path)
    assert out_file is not None
    assert (tmp_path / f"grouped_rule_matrix_{repo_name.lower()}.pdf").exists()


def test_plot_lift_vs_jaccard(tmp_path):
    repo_name = "DummyRepo"
    csv_file = tmp_path / f"rules_{repo_name}.csv"
    make_dummy_rules_csv(csv_file)

    out_file = visualization.plot_lift_vs_jaccard(repo_name, in_dir=tmp_path, out_dir=tmp_path)
    assert out_file is not None
    assert (tmp_path / f"lift_vs_jaccard_{repo_name}.pdf").exists()


def test_plot_lhs_rhs_severity(tmp_path):
    repo_name = "DummyRepo"
    csv_file = tmp_path / f"rules_{repo_name}.csv"
    make_dummy_rules_csv(csv_file)

    out_file = visualization.plot_lhs_rhs_severity(repo_name, in_dir=tmp_path, out_dir=tmp_path, top_n=5)
    assert out_file is not None
    assert (tmp_path / f"lhs_rhs_severity_{repo_name}.pdf").exists()


def test_plot_shared_rules_upset(tmp_path):
    repos = ["Matplotlib", "Sklearn", "Numpy"]
    for repo in repos:
        csv_file = tmp_path / f"rules_{repo}.csv"
        make_dummy_rules_csv(csv_file)

    out_file = visualization.plot_shared_rules_upset(repos, in_dir=tmp_path, out_dir=tmp_path)
    assert out_file is not None
    assert (tmp_path / "shared_rules_upset_errors.pdf").exists()


def test_plot_shared_rules_upset_dynamic(tmp_path):
    # Fake sets of rules for 3 repos
    sets = {
        "RepoA": {"E0001 → E0002", "E0003 → E0004"},
        "RepoB": {"E0001 → E0002"},
        "RepoC": {"E0001 → E0002", "E0005 → E0006"},
    }

    out_file = tmp_path / "shared_rules_upset_dynamic.pdf"
    result = visualization.plot_shared_rules_upset_dynamic(sets, out_file)

    # Assertions
    assert result == out_file
    assert out_file.exists()
    assert out_file.stat().st_size > 0  # file is not empty


def test_plot_lhs_rhs_severity_bubble(tmp_path):
    repo_name = "DummyRepo"
    csv_file = tmp_path / f"rules_new_{repo_name}.csv"

    # Create dummy rules with various RHS severities
    data = [
        {
            "antecedents": "frozenset({'E0001'})",
            "consequents": "frozenset({'E0002'})",
            "Left_Hand_Side": "E0001",
            "Right_Hand_Side": "E0002",
            "support": 0.3,
            "confidence": 0.7,
            "lift": 2.0,
        },
        {
            "antecedents": "frozenset({'C0001'})",
            "consequents": "frozenset({'W0001'})",
            "Left_Hand_Side": "C0001",
            "Right_Hand_Side": "W0001",
            "support": 0.2,
            "confidence": 0.6,
            "lift": 3.0,
        },
        {
            "antecedents": "frozenset({'R0001'})",
            "consequents": "frozenset({'R0002'})",
            "Left_Hand_Side": "R0001",
            "Right_Hand_Side": "R0002",
            "support": 0.4,
            "confidence": 0.8,
            "lift": 2.5,
        },
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    # Run the plotting function
    out_file = visualization.plot_lhs_rhs_severity_bubble(
        repo_name, in_dir=tmp_path, out_dir=tmp_path
    )

    # Assertions
    assert out_file.exists()
    assert out_file.suffix == ".pdf"
    assert "lhs_rhs_severity_bubble" in out_file.name

def test_plot_rule_network(tmp_path):
    import pandas as pd
    from src import visualization

    # Dummy repo
    repo_name = "DummyRepo"
    csv_file = tmp_path / f"rules_new_{repo_name}.csv"

    # Create a minimal dummy rules CSV
    data = [
        {
            "Left_Hand_Side": "E0001",
            "Right_Hand_Side": "E0002",
            "support": 0.3,
            "confidence": 0.7,
            "lift": 2.0,
        },
        {
            "Left_Hand_Side": "E0002",
            "Right_Hand_Side": "E0003",
            "support": 0.2,
            "confidence": 0.6,
            "lift": 1.5,
        },
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    # Run function
    out_file = visualization.plot_rule_network(repo_name, in_dir=tmp_path, out_dir=tmp_path)

    # Assertions
    assert tmp_path.joinpath(f"assoc_network_outdegree_{repo_name}.pdf").exists()
    assert out_file.endswith(".pdf")

def test_plot_asymmetry_matrix(tmp_path):
    import pandas as pd
    from src import visualization

    repo_name = "DummyRepo"
    csv_file = tmp_path / f"rules_{repo_name}.csv"

    # Create dummy 1→1 rules with reverse
    data = [
        {"antecedents": "frozenset({'E0001'})", "consequents": "frozenset({'E0002'})",
         "Left_Hand_Side": "E0001", "Right_Hand_Side": "E0002",
         "support": 0.2, "confidence": 0.8, "lift": 2.0,
         "antecedent support": 0.3, "consequent support": 0.4},
        {"antecedents": "frozenset({'E0002'})", "consequents": "frozenset({'E0001'})",
         "Left_Hand_Side": "E0002", "Right_Hand_Side": "E0001",
         "support": 0.2, "confidence": 0.3, "lift": 1.5,
         "antecedent support": 0.4, "consequent support": 0.3},
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    out_file = visualization.plot_asymmetry_matrix(repo_name, in_dir=tmp_path, out_dir=tmp_path)

    assert out_file is not None
    assert out_file.exists()
    assert out_file.suffix == ".pdf"
