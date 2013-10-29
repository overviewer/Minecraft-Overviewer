#ifndef __BUFFER_H_INCLUDED__
#define __BUFFER_H_INCLUDED__

/* Buffer objects are conveniences for realloc'd dynamic arrays (aka expandable
 * array or arraylist) */

typedef struct {
    void *data;
    /* size of each element in bytes */
    unsigned int element_size;
    /* Number of elements currently in the array */
    unsigned int length;
    /* Number of element slots reserved */
    unsigned int reserved;
} Buffer;

static inline void buffer_init(Buffer *buffer, unsigned int element_size, unsigned int initial_length) {
    buffer->data = NULL;
    buffer->element_size = element_size;
    buffer->length = 0;
    buffer->reserved = initial_length;
}

static inline void buffer_free(Buffer *buffer) {
    if (buffer->data)
        free(buffer->data);
}

static inline void buffer_reserve(Buffer *buffer, unsigned int length) {
    int needs_realloc = 0;
    while (buffer->length + length > buffer->reserved) {
        buffer->reserved *= 2;
        needs_realloc = 1;
    }
    if (buffer->data == NULL)
        needs_realloc = 1;
    
    if (needs_realloc) {
        buffer->data = realloc(buffer->data, buffer->element_size * buffer->reserved);
    }
}

static inline void buffer_append(Buffer *buffer, const void *newdata, unsigned int newdata_length) {
    buffer_reserve(buffer, newdata_length);
    memcpy(buffer->data + (buffer->element_size * buffer->length), newdata, buffer->element_size * newdata_length);
    buffer->length += newdata_length;
}

static inline void buffer_clear(Buffer *buffer) {
    buffer->length = 0;
}


#endif /* __BUFFER_H_INCLUDED__ */
