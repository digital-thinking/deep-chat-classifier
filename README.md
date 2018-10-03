# Deep chat classification

These notebooks have been created during my master thesis about classification of textual dialog data (whatsapp, chats).
The goal of this project was to establish a new architecture for artificial neural networks, which is capable of predicting binary classes of dialogs at an state of a conversation. 

The task can be considered as a special form of goal-oriented dialog state tracking, but with limited output space.
An architecture has been created, which is essentially a model utilizing Sequences of Sequences of word embeddings (SoSoE). 

![alt text](http://digital-thinking.de/wp-content/uploads/2018/07/final.png)

In Keras this architecture looks like the following: 

```python    
    meta_input = Input(shape=(meta_features.values.shape[1],), name='meta_input')
    nlp_seq = Input(shape=(number_of_messages ,seq_length,), name='nlp_input')
    
    # shared layers
    emb = TimeDistributed(Embedding(input_dim=num_features, output_dim=embedding_size,
                    input_length=seq_length, mask_zero=True,
                    input_shape=(seq_length, )))(nlp_seq)    
    x = TimeDistributed(Bidirectional(LSTM(32, dropout=dropout, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01))))(emb)      
    x = Dropout(dropout)(x)
    
    c1 = Conv1D(filter_size, kernel1, kernel_regularizer=regularizers.l2(kernel_reg))(x)
    p1 = GlobalMaxPooling1D()(c1)
    c2 = Conv1D(filter_size, kernel2, kernel_regularizer=regularizers.l2(kernel_reg))(x)
    p2 = GlobalMaxPooling1D()(c2)
    c3 = Conv1D(filter_size, kernel3, kernel_regularizer=regularizers.l2(kernel_reg))(x)
    p3 = GlobalMaxPooling1D()(c3)
    
    x = concatenate([p1, p2, p3, meta_input])    
    x = Dense(classifier_neurons, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid')(x)        
    model = Model(inputs=[nlp_seq, meta_input] , outputs=[x])
   
   ```

## Data

Soon

## Results

Soon
