#pragma once
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

class TaskFreeList {
public:
    static constexpr std::size_t kTaskBytes = 256;
    static constexpr std::size_t kTaskAlign = alignof(std::max_align_t);
    static constexpr std::size_t kChunkBlocks = 1024;

private:
    struct FreeNode {
        FreeNode* next;
    };

    static_assert(kTaskBytes >= sizeof(FreeNode));
    static_assert(kTaskAlign >= alignof(FreeNode));

    struct alignas(kTaskAlign) Block {
        std::byte storage[kTaskBytes];
    };

    FreeNode* head_{nullptr};
    std::vector<std::unique_ptr<Block[]>> chunks_;
    std::mutex mutex_;

    void allocate_chunk() {
        auto chunk = std::make_unique<Block[]>(kChunkBlocks);
        Block* blocks = chunk.get();

        chunks_.push_back(std::move(chunk));

        for (std::size_t i = 0; i < kChunkBlocks; ++i) {
            auto* node = reinterpret_cast<FreeNode*>(static_cast<void*>(blocks[i].storage));
            std::construct_at(node, FreeNode{head_});
            head_ = node;
        }
    }

public:
    TaskFreeList() = default;

    TaskFreeList(const TaskFreeList&) = delete;
    TaskFreeList& operator=(const TaskFreeList&) = delete;
    TaskFreeList(TaskFreeList&&) = delete;
    TaskFreeList& operator=(TaskFreeList&&) = delete;

    [[nodiscard]] static constexpr std::size_t block_size() noexcept {
        return kTaskBytes;
    }

    [[nodiscard]] static constexpr std::size_t block_align() noexcept {
        return kTaskAlign;
    }

    [[nodiscard]] void* acquire() {
        std::lock_guard lock{mutex_};

        if (head_ == nullptr) {
            allocate_chunk();
        }

        FreeNode* node = head_;
        head_ = node->next;
        std::destroy_at(node);
        return node;
    }

    void release(void* ptr) noexcept {
        if (ptr == nullptr) {
            return;
        }

        std::lock_guard lock{mutex_};
        auto* node = static_cast<FreeNode*>(ptr);
        std::construct_at(node, FreeNode{head_});
        head_ = node;
    }

    void reserve(std::size_t block_count) {
        std::lock_guard lock{mutex_};
        while (chunks_.size() * kChunkBlocks < block_count) {
            allocate_chunk();
        }
    }
};
