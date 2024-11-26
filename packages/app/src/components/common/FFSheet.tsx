import {
    BottomSheetBackdrop,
    BottomSheetBackdropProps,
    BottomSheetModal,
    BottomSheetModalProps,
    BottomSheetScrollView,
} from '@gorhom/bottom-sheet'
import { BottomSheetModalMethods } from '@gorhom/bottom-sheet/lib/typescript/types'
import { useTheme } from '@shopify/restyle'
import { forwardRef, PropsWithChildren, Ref } from 'react'
import { Dimensions, View as RNView, ViewProps } from 'react-native'

import { Theme } from '../../theme'
import { FFSafeAreaView, FFView } from './base'

export const SheetHandle = () => (
    <FFView
        backgroundColor="darkGrey"
        height={4}
        width={50}
        borderRadius="full"
        alignSelf="center"
        position="absolute"
        top={10}
    />
)

const SheetBackground = ({ style, ...props }: ViewProps) => {
    const theme = useTheme<Theme>()

    return (
        <RNView
            style={[
                style,
                {
                    backgroundColor: theme.colors.bg,
                    borderColor: theme.colors.bg2,
                    borderWidth: 2,
                    flex: 1,
                    borderTopLeftRadius: 25,
                    borderTopRightRadius: 25,
                },
            ]}
            {...props}
        />
    )
}

const SheetBackdrop = (props: BottomSheetBackdropProps) => (
    <BottomSheetBackdrop pressBehavior="close" style={{ backgroundColor: 'black' }} disappearsOnIndex={-1} {...props} />
)

const FFSheetBase = forwardRef(
    ({ children, ...props }: PropsWithChildren<Partial<BottomSheetModalProps>>, ref: Ref<BottomSheetModalMethods>) => (
        <BottomSheetModal
            ref={ref}
            index={0}
            handleComponent={SheetHandle}
            backgroundComponent={SheetBackground}
            backdropComponent={SheetBackdrop}
            maxDynamicContentSize={Dimensions.get('window').height * 0.9}
            {...props}
        >
            {children}
        </BottomSheetModal>
    )
)

export const FFSheet = forwardRef(
    ({ children, ...props }: PropsWithChildren<Partial<BottomSheetModalProps>>, ref: Ref<BottomSheetModalMethods>) => (
        <FFSheetBase ref={ref} {...props}>
            <FFSafeAreaView paddingHorizontal="sm" paddingVertical="sm" flex={1}>
                {children}
            </FFSafeAreaView>
        </FFSheetBase>
    )
)
export const FFScrollSheet = forwardRef(
    ({ children, ...props }: PropsWithChildren<Partial<BottomSheetModalProps>>, ref: Ref<BottomSheetModalMethods>) => (
        <FFSheetBase ref={ref} {...props}>
            <BottomSheetScrollView>
                <FFSafeAreaView edges={['bottom']} paddingHorizontal="sm" paddingVertical="sm">
                    {children}
                </FFSafeAreaView>
            </BottomSheetScrollView>
        </FFSheetBase>
    )
)
