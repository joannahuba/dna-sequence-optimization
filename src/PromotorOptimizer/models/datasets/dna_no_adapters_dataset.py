from .dna_dataset import DNADataset


class DNADatasetNoAdapters(DNADataset):
    def __init__(self, filepath, is_test=True, header=0):
        # Initialize the base DNADataset (loads self.data and self.mapping)
        super().__init__(filepath, is_test=is_test, header=header)
        
        # Find local adapters globally 
        # self.prefix, self.suffix = find_common_adapters(self.data['sequence'])
        self.prefix_len = 15
        self.suffix_len = 15
        
        # Print info to verify everything is correct
        print(f"--- Dataset Initialized with Adapter Trimming ---")
        print(f"Detected Prefix: '{self.prefix}' ({self.prefix_len} bp)")
        print(f"Detected Suffix: '{self.suffix}' ({self.suffix_len} bp)")
        
    def __getitem__(self, idx):
        # Get the original output from the parent class
        # (Handling one-hot encoding and tensor conversion there)
        result = super().__getitem__(idx)
        
        if self.is_test:
            # result is (seq_id, sequence_tensor)
            seq_id, sequence = result
            # Trim the sequence: [channels, start:end]
            end_idx = sequence.shape[1] - self.suffix_len if self.suffix_len > 0 else sequence.shape[1]
            trimmed_seq = sequence[:, self.prefix_len : end_idx]
            return seq_id, trimmed_seq
        else:
            # result is (sequence_tensor, ratio, active)
            sequence, ratio, active = result
            # Trim the sequence: [channels, start:end]
            end_idx = sequence.shape[1] - self.suffix_len if self.suffix_len > 0 else sequence.shape[1]
            trimmed_seq = sequence[:, self.prefix_len : end_idx]
            return trimmed_seq, ratio, active
        
##########################
# Helpers 
##########################


def find_common_adapters(sequences):
    # Convert series to list for faster iteration
    seq_list = sequences.tolist()
    if not seq_list:
        return "", ""

    # 1. Find common prefix (from the front)
    common_prefix = ""
    min_len = min(len(s) for s in seq_list)
    
    for i in range(min_len):
        # Get the i-th character from all sequences
        chars_at_pos = set(s[i] for s in seq_list)
        # If all sequences have the same character at this position
        if len(chars_at_pos) == 1:
            common_prefix += seq_list[0][i]
        else:
            break
            
    # 2. Find common suffix (from the back)
    common_suffix = ""
    for i in range(1, min_len + 1):
        # Get the i-th character from the end
        chars_at_pos = set(s[-i] for s in seq_list)
        if len(chars_at_pos) == 1:
            common_suffix = seq_list[0][-i] + common_suffix
        else:
            break
            
    return common_prefix, common_suffix