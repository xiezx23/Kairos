import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--full", type=bool, default=False, 
                    help="Use full precision model or not.")
parser.add_argument("--model_type", type=str, default='qwen', choices=['llama', 'qwen'], 
                    help="Model Type: qwen / llama")
parser.add_argument("--load_quant_model", type=str, default=None, 
                    help="Path to load quantized model.")
parser.add_argument("--dump_quant_model", type=str, default=None, 
                    help="Path to save quantized model.")
parser.add_argument("--eval_ppl", type=bool, default=False, 
                    help="Enable evaluation mode or not.")
parser.add_argument("--dump_calibration", type=bool, default=False, 
                    help="Whether to dump calibration data.")
parser.add_argument("--test_case", type=str, default='batch_decode',
                    help="Test the prefill or batch_decode phase")
parser.add_argument("--quant_type", type=str, default=None,
                    help="Quant Type: 'W8A8Linear', 'W4A16Linear', 'W4A8Linear', 'DynamicLinear'")