from experiments.module2.pipeline import run_module2_sweep


if __name__ == "__main__":
    print(
        run_module2_sweep(
            parameter_name="T",
            run_name="module2_T_sweep",
        )
    )
