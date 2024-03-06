def CreateDataLoader(
        datafolder,
        dataroot='./dataset',
        dataset_mode='2afc',
        load_size=64,
        batch_size=1,
        serial_batches=True,
        nThreads=4,
        use_cache=False
        ):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    # print(data_loader.name())
    data_loader.initialize(
        datafolder,
        dataroot=dataroot+'/'+dataset_mode,
        dataset_mode=dataset_mode,
        load_size=load_size,
        batch_size=batch_size,
        serial_batches=serial_batches,
        nThreads=nThreads,
        use_cache=use_cache)
    return data_loader
