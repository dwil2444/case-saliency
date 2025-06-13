# Clean (optional) - add any clean-up commands if necessary
clean:
	rm -rf *.log *.out *.err *.pdf
	find . -type d -name "*__pycache__*" | xargs -I {} rm -rf {}
	find -maxdepth 2 -type f -name "*.png" | xargs -I {} rm -rf {} \;
	rm -rf notebooks/.ipynb_checkpoints *.csv
	rm -rf .dump
	rm -rf .rqone .rqtwo .ablation
	rm -rf imgs/*