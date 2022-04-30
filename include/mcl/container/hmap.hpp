// This file is part of the mcl project.
// Copyright (c) 2022 merryhime
// SPDX-License-Identifier: MIT

#pragma once

#include <bit>
#include <utility>

#include "mcl/assert.hpp"
#include "mcl/bitsizeof.hpp"
#include "mcl/hint/assume.hpp"
#include "mcl/macro/architecture.hpp"
#include "mcl/stdint.hpp"

#if defined(MCL_ARCHITECTURE_ARM64)
#    include "arm_neon.h"
#endif

namespace mcl {

template<typename K, typename T>
class hmap;

namespace detail {

/// if MSB is 0, this is a full slot. remaining 7 bits is a partial hash of the key.
/// if MSB is 1, this is a non-full slot.
enum class meta_byte : u8 {
    empty = 0xff,
    tombstone = 0x80,
    end_sentinel = 0x88,
};

inline bool is_full(meta_byte mb)
{
    return (static_cast<u8>(mb) & 0x80) == 0;
}

inline meta_byte meta_byte_from_hash(size_t hash)
{
    return static_cast<meta_byte>(hash >> (bitsizeof<size_t> - 7));
}

inline size_t group_index_from_hash(size_t hash, size_t group_index_mask)
{
    return hash & group_index_mask;
}

#if defined(MCL_ARCHITECTURE_ARM64)

struct meta_byte_group {
    meta_byte_group(meta_byte* ptr)
        : data{vld1q_u8(reinterpret_cast<u8*>(ptr))}
    {}

    uint64x2_t match(meta_byte cmp)
    {
        return vreinterpretq_u64_u8(vandq_u8(vceqq_u8(data,
                                                      vdupq_n_u8(static_cast<u8>(cmp))),
                                             vdupq_n_u8(0x80)));
    }

    uint64x2_t match_empty_or_tombstone()
    {
        return vreinterpretq_u64_u8(vandq_u8(data,
                                             vdupq_n_u8(0x80)));
    }

    bool is_any_empty()
    {
        static_assert(meta_byte::empty == static_cast<meta_byte>(0xff), "empty must be maximal u8 value");
        return vmaxvq_u8(data) == 0xff;
    }

    uint8x16_t data;
};

#    define MCL_HMAP_MATCH_META_BYTE_GROUP(MATCH, ...)                                                             \
        {                                                                                                          \
            const uint64x2_t match_result = MATCH;                                                                 \
                                                                                                                   \
            for (u64 match_result_v{match_result[0]}; match_result_v != 0; match_result_v &= match_result_v - 1) { \
                const size_t match_index = std::countr_zero(match_result_v) / 8;                                   \
                __VA_ARGS__                                                                                        \
            }                                                                                                      \
                                                                                                                   \
            for (u64 match_result_v{match_result[1]}; match_result_v != 0; match_result_v &= match_result_v - 1) { \
                const size_t match_index = 8 + std::countr_zero(match_result_v) / 8;                               \
                __VA_ARGS__                                                                                        \
            }                                                                                                      \
        }

#else
#    error "TODO: Generic implementation"
#endif

template<typename ValueType>
union slot_union {
    slot_union() {}
    ValueType value;
};

}  // namespace detail

template<typename K, typename T>
class hmap_iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = std::pair<K, T>;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference = value_type&;
    using const_reference = const value_type&;

    hmap_iterator() = default;
    hmap_iterator(const hmap_iterator& other) = default;
    hmap_iterator& operator=(const hmap_iterator& other) = default;

    hmap_iterator& operator++()
    {
        if (mb_ptr == nullptr)
            return *this;

        ++mb_ptr;
        ++slot_ptr;

        skip_empty_or_tombstone();

        return *this;
    }
    hmap_iterator operator++(int)
    {
        hmap_iterator it(*this);
        ++*this;
        return it;
    }

    bool operator==(const hmap_iterator& other) const
    {
        return std::tie(mb_ptr, slot_ptr) == std::tie(other.mb_ptr, other.slot_ptr);
    }
    bool operator!=(const hmap_iterator& other) const
    {
        return !operator==(other);
    }

    reference operator*() const
    {
        return static_cast<reference>(slot_ptr->value);
    }
    pointer operator->() const
    {
        return std::addressof(operator*());
    }

private:
    friend class hmap<K, T>;

    using slot_type = detail::slot_union<value_type>;

    hmap_iterator(detail::meta_byte* mb_ptr, slot_type* slot_ptr)
        : mb_ptr{mb_ptr}, slot_ptr{slot_ptr}
    {
        ASSUME(mb_ptr != nullptr);
        ASSUME(slot_ptr != nullptr);
    }

    void skip_empty_or_tombstone()
    {
        if (!mb_ptr)
            return;

        while (!is_full(*mb_ptr)) {
            ++mb_ptr;
            ++slot_ptr;

            if (*mb_ptr == detail::meta_byte::end_sentinel) {
                mb_ptr = nullptr;
                slot_ptr = nullptr;
                return;
            }
        }
    }

    detail::meta_byte* mb_ptr = nullptr;
    slot_type* slot_ptr = nullptr;
};

template<typename KeyType, typename MappedType>
class hmap {
public:
    using key_type = KeyType;
    using mapped_type = MappedType;
    using value_type = std::pair<key_type, mapped_type>;
    using iterator = hmap_iterator<key_type, mapped_type>;

private:
    using slot_type = detail::slot_union<value_type>;
    static_assert(!std::is_reference_v<key_type>);
    static_assert(!std::is_reference_v<mapped_type>);

    static constexpr size_t group_size = 16;

public:
    hmap()
    {
        initialize_members(1);
    }

    ~hmap()
    {
        for (auto iter = begin(); iter != end(); ++iter) {
            iter->~value_type();
        }
    }

    iterator begin()
    {
        iterator result{get_iterator_at(0)};
        result.skip_empty_or_tombstone();
        return result;
    }
    iterator end()
    {
        return {};
    }

    iterator find(const key_type& key)
    {
        const size_t hash = std::hash<key_type>{}(key);
        const detail::meta_byte mb = detail::meta_byte_from_hash(hash);

        size_t group_index = detail::group_index_from_hash(hash, group_index_mask);

        while (true) {
            detail::meta_byte_group g{mbs.get() + group_index * group_size};

            MCL_HMAP_MATCH_META_BYTE_GROUP(g.match(mb), {
                const size_t item_index{group_index * group_size + match_index};

                if (slots[item_index].value.first == key) [[likely]] {
                    return get_iterator_at(item_index);
                }
            });

            if (g.is_any_empty()) [[likely]] {
                return {};
            }

            group_index = (group_index + 1) & group_index_mask;
        }
    }

    template<typename K, typename... Args>
    std::pair<iterator, bool> try_emplace(K&& k, Args&&... args)
    {
        auto [item_index, item_found] = find_key_or_empty_slot(k);
        if (!item_found) {
            new (&slots[item_index].value) value_type(
                std::piecewise_construct,
                std::forward_as_tuple(std::forward<K>(k)),
                std::forward_as_tuple(std::forward<Args>(args)...));
        }
        return {get_iterator_at(item_index), !item_found};
    }

    template<typename K, typename V>
    std::pair<iterator, bool> insert_or_assign(K&& k, V&& v)
    {
        auto [item_index, item_found] = find_key_or_empty_slot(k);
        if (item_found) {
            slots[item_index].value.second = std::forward<V>(v);
        } else {
            new (&slots[item_index].value) value_type(
                std::forward<K>(k),
                std::forward<V>(v));
        }
        return {get_iterator_at(item_index), !item_found};
    }

    template<typename K = key_type>
    mapped_type& operator[](K&& k)
    {
        return try_emplace(std::forward<K>(k)).first->second;
    }

private:
    iterator get_iterator_at(size_t item_index)
    {
        return {mbs.get() + item_index, slots.get() + item_index};
    }

    std::pair<size_t, bool> find_key_or_empty_slot(const key_type& key)
    {
        const size_t hash = std::hash<key_type>{}(key);
        const detail::meta_byte mb = detail::meta_byte_from_hash(hash);

        size_t group_index = detail::group_index_from_hash(hash, group_index_mask);

        while (true) {
            detail::meta_byte_group g{mbs.get() + group_index * group_size};

            MCL_HMAP_MATCH_META_BYTE_GROUP(g.match(mb), {
                const size_t item_index{group_index * group_size + match_index};

                if (slots[item_index].value.first == key) [[likely]] {
                    return {item_index, true};
                }
            });

            if (g.is_any_empty()) [[likely]] {
                return {find_empty_slot_to_insert(hash), false};
            }

            group_index = (group_index + 1) & group_index_mask;
        }
    }

    size_t find_empty_slot_to_insert(const size_t hash)
    {
        if (empty_slots == 0) [[unlikely]] {
            grow_and_rehash();
        }

        size_t group_index = detail::group_index_from_hash(hash, group_index_mask);

        while (true) {
            detail::meta_byte_group g{mbs.get() + group_index * group_size};

            MCL_HMAP_MATCH_META_BYTE_GROUP(g.match_empty_or_tombstone(), {
                const size_t item_index{group_index * group_size + match_index};

                if (mbs[item_index] == detail::meta_byte::empty) [[likely]] {
                    --empty_slots;
                }

                mbs[item_index] = detail::meta_byte_from_hash(hash);

                return item_index;
            });

            group_index = (group_index + 1) & group_index_mask;
        }
    }

    void grow_and_rehash()
    {
        const size_t new_group_count = 2 * (group_index_mask + 1);

        pow2_resize(new_group_count);
    }

    void pow2_resize(size_t new_group_count)
    {
        auto iter = begin();

        const auto old_mbs = std::move(mbs);
        const auto old_slots = std::move(slots);

        initialize_members(new_group_count);

        for (; iter != iterator{}; ++iter) {
            const size_t hash = std::hash<key_type>{}(iter->first);
            const size_t item_index = find_empty_slot_to_insert(hash);

            new (&slots[item_index].value) value_type(std::move(iter.slot_ptr->value));
            iter.slot_ptr->value.~value_type();
        }
    }

    void initialize_members(size_t group_count)
    {
        // DEBUG_ASSERT(group_count != 0 && std::ispow2(group_count));

        group_index_mask = group_count - 1;
        empty_slots = group_count * group_size * 7 / 8;
        mbs = std::unique_ptr<detail::meta_byte[]>{new (std::align_val_t(group_size)) detail::meta_byte[group_count * group_size + 1]};
        slots = std::unique_ptr<slot_type[]>{new slot_type[group_count * group_size]};

        std::memset(mbs.get(), static_cast<int>(detail::meta_byte::empty), group_count * group_size);
        mbs[group_count * group_size] = detail::meta_byte::end_sentinel;
    }

    size_t group_index_mask;
    size_t empty_slots;
    std::unique_ptr<detail::meta_byte[]> mbs;
    std::unique_ptr<slot_type[]> slots;
};

}  // namespace mcl
