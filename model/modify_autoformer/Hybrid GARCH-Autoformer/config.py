TIERS_CONFIG = {
    'Tier_1_Miniaturized': {
        'd_model': 32,
        'e_layers': 2,
        'n_heads': 4,
        'd_ff': 128,
        'dropout': 0.3
    },
    'Tier_2_Standard': {
        'd_model': 128,
        'e_layers': 3,
        'n_heads': 8,
        'd_ff': 512,
        'dropout': 0.2
    },
    'Tier_3_Large': {
        'd_model': 512,
        'e_layers': 6,
        'n_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1
    }
}
